from __future__ import annotations
import copy
import torch
import numpy as np
import nibabel as nib
from monai.transforms import Orientation
from monai.data import MetaTensor


def reorient_to_ras(data: np.ndarray, affine: np.ndarray) -> MetaTensor:
    """Wraps data as a MetaTensor and reorients it to RAS. Seeds 'original_affine' on
    the meta dict (mirroring what MONAI's LoadImaged tracks) so reorient_from_ras()
    can invert it later using the same pattern already established in
    clopa/adaptation/io_operations.py's WriteImage and IS_Validate's
    write_image_utils/post.py."""
    affine_t = torch.as_tensor(affine)
    meta = MetaTensor(
        torch.from_numpy(data.astype(np.float32)),
        meta={'affine': affine_t, 'original_affine': affine_t},
    )
    return Orientation(axcodes='RAS')(meta)


def reorient_from_ras(pred: torch.Tensor, ras_meta: MetaTensor) -> np.ndarray:
    """Invert RAS reorientation on a prediction, following the same pattern as
    clopa/adaptation/io_operations.py's WriteImage / IS_Validate's
    write_image_utils/post.py: substitute the prediction's values into a duplicate
    of the reference (image) MetaTensor, then reapply Orientation forward targeting
    the tracked original axcodes, with trace recording disabled. Dtype is cast back
    explicitly at the end (matching write_itk()'s .astype(dtype) convention) since
    the transform pipeline itself doesn't preserve it — duplicate.array inherits the
    reference's dtype, not the prediction's."""
    duplicate = copy.deepcopy(ras_meta)
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
    duplicate.array = pred_np

    orig_affine = duplicate.meta.get('original_affine')
    if orig_affine is None:
        raise Exception('Failed to invert orientation — original_affine is not on the image header')
    orig_axcodes = nib.orientations.aff2axcodes(
        orig_affine.numpy() if isinstance(orig_affine, torch.Tensor) else orig_affine
    )
    inverse_transform = Orientation(axcodes=orig_axcodes)
    with inverse_transform.trace_transform(False):
        result = inverse_transform(duplicate)

    return result.numpy().astype(pred_np.dtype)


def compute_ras_ornt(affine: np.ndarray) -> np.ndarray:
    """Computes the same axis permutation + flip transform reorient_to_ras() applies to the
    image — identical nibabel calls to what MONAI's Orientation transform makes internally
    (io_orientation + axcodes2ornt('RAS') + ornt_transform) — so interaction coordinates,
    recorded in the image's original pre-reorientation axis order, can be mapped into the same
    RAS-oriented frame the image itself gets put into."""
    src = nib.io_orientation(affine)
    dst = nib.orientations.axcodes2ornt('RAS')
    return nib.orientations.ornt_transform(src, dst)


def remap_point_to_ras(point, ornt: np.ndarray, orig_shape) -> list:
    """Maps a single (x0, x1, ..., xn) coordinate from the image's original axis order into
    the order/flip reorient_to_ras() put the image into. Orientation transforms here are
    always signed permutations — nib.orientations can only represent axis swaps plus per-axis
    sign flips, never arbitrary rotation — so this is a direct index-level transcription of
    nibabel's apply_orientation() (flip, then permute), applied to one coordinate instead of a
    whole array; the two are mathematically identical, this is just far cheaper per point.
    Flip uses the *original* array's size along each axis, since flip precedes permute in the
    underlying transform. Reusable per-point for scribble/lasso too, once those are converted
    to a point-set representation — not yet decided."""
    flipped = [
        (orig_shape[ax] - 1 - point[ax]) if ornt[ax, 1] == -1 else point[ax]
        for ax in range(len(point))
    ]
    axes = np.argsort(ornt[:, 0])
    return [flipped[ax] for ax in axes]


class CLoPASession:
    """
    Adapter that accumulates napari interactions and translates them into
    InferApp request dicts. InferApp.__call__() is the sole inference path —
    this class never calls the raw session's prediction methods directly.
    """

    _PROMPT_TYPES = ('points', 'bboxes', 'scribbles', 'lasso')

    def __init__(self, infer_app, dataset_level_schema: dict):
        self.app = infer_app
        self.session = infer_app.session
        self.dataset_level_schema = dataset_level_schema
        self._image_meta = {}
        self._is_init = True
        self._display_buffer = None
        self.sample_level_schema = None   # set once per set_image() call, see set_image()
        self._reset_accumulator()

    @property
    def semantic_id_dict(self):
        return self.dataset_level_schema['segmentation_task_schema']['semantic_id_dict']

    def _reset_accumulator(self):
        # scribbles/lasso: not implemented yet — add_scribble_interaction() /
        # add_lasso_interaction() raise NotImplementedError. Event-handling mechanics
        # (how a single stroke/shape reaches this class) are documented in
        # napari-clopa/docs/interaction-event-flow.md. Still open: storage shape
        # (list of full masks vs. progressive merge) and InferApp conversion format
        # (binary_place_interactions() doesn't support scribbles/lasso yet) — see
        # plan Step 2c.
        self._accumulated = {
            'points': [],
            'points_labels': [],
            'bboxes': [],
            'bboxes_labels': [],
            'scribbles': None,
            'scribbles_labels': None,
            'lasso': None,
            'lasso_labels': None,
        }

    def _label_code(self, include_interaction: bool) -> int:
        if include_interaction:
            fg_key = next(k for k in self.semantic_id_dict if k != 'background')
            return self.semantic_id_dict[fg_key]
        return self.semantic_id_dict['background']

    def _check_single_prompt_type(self, incoming: str):
        active = [
            ptype for ptype in self._PROMPT_TYPES
            if ptype != incoming and self._accumulated[ptype]
        ]
        if active:
            raise ValueError(
                f"Cannot accumulate a '{incoming}' interaction while unflushed "
                f"'{active[0]}' interactions are pending — InferApp only accepts one "
                f"prompt type per request. Predict (flush) before switching prompt types."
            )

    # --- Image setup + RAS handling ---

    def set_image(self, data: np.ndarray, props: dict):
        if 'task_channels' not in props:
            raise ValueError("props must include 'task_channels' (per-case, see export config)")

        affine_in = props.get('affine')
        if affine_in is None:
            raise ValueError("props must include 'affine'")
        affine = affine_in.affine_matrix if hasattr(affine_in, 'affine_matrix') else np.asarray(affine_in)

        ras_meta = reorient_to_ras(data, affine)
        self._image_meta = {
            'affine': ras_meta.affine,
            'tensor': ras_meta,
            'ornt': compute_ras_ornt(affine),
            'orig_shape': data.shape[1:],
        }
        # sample_level_schema only updates once _image_meta has successfully been built
        # above — if reorient_to_ras()/compute_ras_ornt() raises on a second set_image()
        # call (bad affine on a re-load), neither updates, so the two never drift out of
        # sync with each other on a partial failure. Built once here, not reassembled on
        # every _build_request() call — nothing in it changes between flushes for the
        # same loaded image, only when a new one is set.
        self.sample_level_schema = {
            'data_schema': {'task_channels': props['task_channels']},
            'segmentation_task_schema': {'semantic_id_dict': self.semantic_id_dict},
        }
        # Not calling self.app.load_new_image() here — InferApp.binary_subject_prep()
        # already calls it internally for every IS_interactive_init request, and
        # _is_init below guarantees the next request is one. Calling it here too
        # would just be redundant, thrown-away work.

        self._is_init = True
        self._reset_accumulator()

    # --- Pass-throughs ---

    @property
    def supported_interactions(self):
        if 'permitted_prompts' not in self.app.app_params:
            raise ValueError(
                "InferApp.app_params is missing 'permitted_prompts' — every InferApp "
                "construction branch sets this, so its absence means something is "
                "genuinely wrong, not something to silently default around."
            )
        permitted = self.app.app_params['permitted_prompts']
        return {
            'points': 'points' in permitted,
            'bbox2d': 'bboxes' in permitted,
            'scribble': 'scribbles' in permitted,
            'lasso': 'lasso' in permitted,
        }

    @property
    def preferred_scribble_thickness(self):
        return self.session.preferred_scribble_thickness

    def set_target_buffer(self, buf):
        self._display_buffer = buf
        self.session.set_target_buffer(buf)

    def reset_interactions(self):
        self._reset_accumulator()
        self.session.reset_interactions()
        # self.session.target_buffer gets reallocated on every IS_interactive_init request
        # (InferApp.load_new_image()), so it's no longer the same object as
        # self._display_buffer by the time an object has had even one prediction — the
        # raw-session reset above only zeroes the (by-then-orphaned) reallocated buffer,
        # not what's actually on screen. Zero the display buffer explicitly so the label
        # layer doesn't keep showing the last prediction until the next flush overwrites it.
        if self._display_buffer is not None:
            if isinstance(self._display_buffer, np.ndarray):
                self._display_buffer.fill(0)
            elif isinstance(self._display_buffer, torch.Tensor):
                self._display_buffer.zero_()
        self._is_init = True

    def reset_pending_interactions(self):
        # Only clears interactions placed since the last flush — never touches the raw
        # session (self.session), so the object's already-predicted history and
        # target_buffer are untouched. Distinct from reset_interactions(), which wipes
        # the whole object and forces the next flush back to IS_interactive_init.
        self._reset_accumulator()

    def set_do_autozoom(self, do_autozoom, max_num_patches=None):
        self.session.set_do_autozoom(do_autozoom, max_num_patches)

    def close(self):
        self.app.close()

    # --- Accumulation methods ---
    #
    # One call per finished prompt instance (point/bbox committed, or a scribble/lasso
    # shape once implemented) — not once per batch, and not only at inference time.
    # That one-call-per-instance contract is why points/bboxes accumulate as lists
    # appended to per call. Full event chain from napari mouse event to this call:
    # napari-clopa/docs/interaction-event-flow.md.

    def add_point_interaction(
        self,
        coordinates: tuple,
        include_interaction: bool,
        run_prediction: bool = True,
    ):
        self._check_single_prompt_type('points')
        ras_coord = remap_point_to_ras(
            coordinates, self._image_meta['ornt'], self._image_meta['orig_shape']
        )
        self._accumulated['points'].append(ras_coord)
        self._accumulated['points_labels'].append(self._label_code(include_interaction))
        if run_prediction:
            self._flush()

    def add_bbox_interaction(
        self,
        bbox_coords: list,
        include_interaction: bool,
        run_prediction: bool = True,
    ):
        self._check_single_prompt_type('bboxes')
        # bbox_coords is [[x_min, x_max], [y_min, y_max], [z_min, z_max]] — remap each corner
        # separately, since a flip can turn what was the min corner into the max corner along
        # a given axis, then re-sort per axis to recover a valid [min, max] pair. InferApp
        # (app.py's box[0, i] / box[0, i+3] indexing) requires a flat 6-element
        # [x_min, y_min, z_min, x_max, y_max, z_max] layout per box, not nested pairs.
        corner_a = [axis[0] for axis in bbox_coords]
        corner_b = [axis[1] for axis in bbox_coords]
        ras_a = remap_point_to_ras(corner_a, self._image_meta['ornt'], self._image_meta['orig_shape'])
        ras_b = remap_point_to_ras(corner_b, self._image_meta['ornt'], self._image_meta['orig_shape'])
        ras_pairs = [sorted(pair) for pair in zip(ras_a, ras_b)]
        ras_bbox = [lo for lo, hi in ras_pairs] + [hi for lo, hi in ras_pairs]
        self._accumulated['bboxes'].append(ras_bbox)
        self._accumulated['bboxes_labels'].append(self._label_code(include_interaction))
        if run_prediction:
            self._flush()

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
    ):
        raise NotImplementedError(
            "Scribble interactions are not yet supported — InferApp.binary_place_interactions() "
            "has no scribble handling, and storage/conversion format is undecided (see plan Step 2c). "
            "Once a point-set representation is chosen, remap_point_to_ras() applies per point."
        )

    def add_lasso_interaction(
        self,
        lasso_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
    ):
        raise NotImplementedError(
            "Lasso interactions are not yet supported — InferApp.binary_place_interactions() "
            "has no lasso handling, and storage/conversion format is undecided (see plan Step 2c). "
            "Once a point-set representation is chosen, remap_point_to_ras() applies per point."
        )

    def add_initial_seg_interaction(self, mask: np.ndarray, run_prediction: bool = True):
        # TODO: "start from an existing segmentation" — conceptually a form of auto-init,
        # but InferApp has no clean backend path for it (autoseg_infer_bool is hardcoded
        # False/unsupported, and even when set the IS_autoseg branch in app.py never
        # actually raises its NotImplementedError and falls through to an undefined
        # is_state). Do not hack this in on the napari side — InferApp is the shared
        # inference contract also used by IS-Validate's direct calls, and a napari-only
        # workaround would make the two callers diverge on what "auto init" means. See
        # plan Known Limitations item 3. Proper support needs backend-level design first.
        raise NotImplementedError(
            "Initializing from an existing segmentation is not yet supported — "
            "InferApp has no backend path for it (see plan Known Limitations item 3)."
        )

    # --- Request building + prediction ---

    def _build_request(self):
        if self.sample_level_schema is None:
            raise ValueError('No image loaded — set_image() must run before building a request.')

        interactions = {}
        labels = {}

        if self._accumulated['points']:
            interactions['points'] = [
                torch.tensor([pt], dtype=torch.float32) for pt in self._accumulated['points']
            ]
            labels['points_labels'] = [
                torch.tensor([lbl], dtype=torch.int64) for lbl in self._accumulated['points_labels']
            ]

        if self._accumulated['bboxes']:
            interactions['bboxes'] = [
                torch.tensor([bbox], dtype=torch.float32) for bbox in self._accumulated['bboxes']
            ]
            labels['bboxes_labels'] = [
                torch.tensor([lbl], dtype=torch.int64) for lbl in self._accumulated['bboxes_labels']
            ]

        # scribble and lasso: conversion TBD, format TBD (see plan Step 2c) — once the
        # API format is finalised, values are placed here following the same
        # list-of-per-interaction-tensors convention as points/bboxes above

        return {
            'sample_level_schema': self.sample_level_schema,
            # 'IS_autoseg' (no-prompt inference) is a third valid infer_mode InferApp
            # defines but never reachable here — deferred, see add_initial_seg_interaction().
            'infer_mode': 'IS_interactive_init' if self._is_init else 'IS_interactive_edit',
            'i_state': {
                'interaction_torch_format': {
                    'interactions': interactions,
                    'interactions_labels': labels,
                },
            },
            'image': {
                'metatensor': self._image_meta['tensor'],
                'meta_dict': {'affine': self._image_meta['affine']},
            },
        }

    def _flush(self):
        request = self._build_request()
        output = self.app(request)

        returned_affine = output['pred']['meta_dict']['affine']
        sent_affine = request['image']['meta_dict']['affine']
        if not torch.allclose(returned_affine.float(), sent_affine.float(), atol=1e-5):
            raise ValueError(
                'InferApp returned an affine that does not match what was sent — '
                'possible metadata corruption in the request/response round-trip.'
            )

        pred = output['pred']['metatensor'].squeeze(0)
        original = reorient_from_ras(pred, self._image_meta['tensor'])

        if self._display_buffer is not None:
            np.copyto(self._display_buffer, original)

        self._reset_accumulator()
        self._is_init = False

    def _predict(self):
        if not any(self._accumulated[ptype] for ptype in self._PROMPT_TYPES):
            raise ValueError('No pending interactions to run — place a prompt before running.')
        return self._flush()
