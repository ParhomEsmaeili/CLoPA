from __future__ import annotations
from typing import Any, Optional
import torch
import numpy as np


class CLoPASession:
    """
    Session-compatible wrapper that accumulates napari interactions and
    routes prediction through InferApp for canonical prompt processing.

    Exposes the same interface as nnInteractiveInferenceSession so the
    napari widget can use it as a drop-in replacement for self.session.
    Methods that don't need InferApp processing are passed through to the
    raw session directly.
    """

    def __init__(self, session, app_params=None):
        self.session = session
        self._app_params = app_params or {}
        self._reset_accumulator()

    def _reset_accumulator(self):
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

    # --- Pass-through methods (no InferApp needed) ---

    @property
    def preferred_scribble_thickness(self):
        return self.session.preferred_scribble_thickness

    @property
    def supported_interactions(self):
        permitted = self._app_params.get('permitted_prompts', ())
        return {
            'points': 'points' in permitted,
            'bbox2d': 'bboxes' in permitted,
            'scribble': 'scribbles' in permitted,
            'lasso': 'lasso' in permitted,
        }

    @property
    def license(self):
        return getattr(self.session, 'license', None)

    def set_image(self, image: np.ndarray, image_properties: dict = None):
        self.session.set_image(image, image_properties)

    def set_target_buffer(self, target_buffer):
        self.session.set_target_buffer(target_buffer)

    def set_do_autozoom(self, do_autozoom: bool, max_num_patches: Optional[int] = None):
        self.session.set_do_autozoom(do_autozoom, max_num_patches)

    def reset_interactions(self):
        self._reset_accumulator()
        self.session.reset_interactions()

    def undo(self):
        return self.session.undo()

    # --- Accumulation methods ---

    def add_point_interaction(
        self,
        coordinates: tuple,
        include_interaction: bool,
        run_prediction: bool = True,
        skip_interaction_decay: bool = False,
    ):
        self._accumulated['points'].append(coordinates)
        label = 1 if include_interaction else 0
        self._accumulated['points_labels'].append(label)
        if run_prediction:
            self._flush()

    def add_bbox_interaction(
        self,
        bbox_coords: list,
        include_interaction: bool,
        run_prediction: bool = True,
        skip_interaction_decay: bool = False,
    ):
        self._accumulated['bboxes'].append(bbox_coords)
        label = 1 if include_interaction else 0
        self._accumulated['bboxes_labels'].append(label)
        if run_prediction:
            self._flush()

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        skip_interaction_decay: bool = False,
    ):
        self._accumulated['scribbles'] = scribble_image
        self._accumulated['scribbles_labels'] = 1 if include_interaction else 0
        if run_prediction:
            self._flush()

    def add_lasso_interaction(
        self,
        lasso_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        skip_interaction_decay: bool = False,
    ):
        self._accumulated['lasso'] = lasso_image
        self._accumulated['lasso_labels'] = 1 if include_interaction else 0
        if run_prediction:
            self._flush()

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False):
        self._initial_seg = initial_seg
        if run_prediction:
            self._flush()

    # --- Prediction ---

    def _flush(self):
        """Place accumulated interactions on the session and run prediction."""
        for coords, label in zip(self._accumulated['points'], self._accumulated['points_labels']):
            self.session.add_point_interaction(
                tuple(coords), bool(label), run_prediction=False, skip_interaction_decay=True
            )
        for bbox, label in zip(self._accumulated['bboxes'], self._accumulated['bboxes_labels']):
            self.session.add_bbox_interaction(
                bbox, bool(label), run_prediction=False, skip_interaction_decay=True
            )
        self.session.new_interaction_centers = [self.session.new_interaction_centers[-1]]
        self.session.new_interaction_zoom_out_factors = [self.session.new_interaction_zoom_out_factors[-1]]
        self.session._predict()
        self._reset_accumulator()

    def _predict(self):
        return self._flush()

    # --- Internal helpers ---
