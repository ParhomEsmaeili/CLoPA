# Interaction event flow: `CLoPASession` to `InferApp`

How a single interaction becomes an `InferApp` inference call, once it reaches
`CLoPASession`. This covers the half of the pipeline that lives in this repo. For how a
napari mouse interaction turns into a `CLoPASession.add_X_interaction()` call in the first
place — Qt mouse events, napari layer event emission, the widget's dispatch logic — see
[napari-clopa's `docs/interaction-event-flow.md`](https://github.com/ParhomEsmaeili/napari-clopa/blob/main/docs/interaction-event-flow.md).
For the complete, single-page version of both halves, see the Daneshvar wiki page
`wiki/sources/napari-clopa-interaction-event-flow.md`.

## Entry point: `add_X_interaction()`

The widget calls these one at a time, per finished interaction — not once per batch, and not
only at inference time (see the napari-clopa doc for why: each committed stroke/point/box
triggers exactly one call).

```python
def add_point_interaction(self, coordinates, include_interaction, run_prediction=True, ...):
    self._check_single_prompt_type('points')
    self._accumulated['points'].append(coordinates)
    self._accumulated['points_labels'].append(self._label_code(include_interaction))
    if run_prediction:
        self._flush()
```

- `_check_single_prompt_type()` blocks mixing prompt types in one unflushed batch — you can
  accumulate several points, or several bboxes, but not both before a flush. `InferApp` only
  accepts one prompt type per request.
- `run_prediction` (fed from the widget's autorun checkbox, default off) decides whether this
  call also triggers `_flush()` immediately, or just accumulates.

Scribbles and lasso currently raise `NotImplementedError` here instead of accumulating —
`_build_request()` never consumed that data anyway (see below), and
`InferApp.binary_place_interactions()` doesn't support them yet, so silently accepting and
discarding the data would be misleading.

## `_build_request()`: accumulator → `InferApp` request dict

Bundles everything in `self._accumulated` since the last flush into one request:

```python
{
    'sample_level_schema': {
        'data_schema': {'task_channels': task_channels},
        'segmentation_task_schema': {'semantic_id_dict': semantic_id_dict},
    },
    'infer_mode': 'IS_interactive_init' if self._is_init else 'IS_interactive_edit',
    'i_state': {'interaction_torch_format': {'interactions': ..., 'interactions_labels': ...}},
    'image': {'metatensor': ..., 'meta_dict': {'affine': ...}},
}
```

Key contract: `InferApp.binary_place_interactions()` expects **lists of individually-shaped
per-interaction tensors** — `(1, 3)` per point, `(1, 6)` per bbox, `(1,)` per label — not one
stacked tensor, because it calls `torch.cat()` and indexes with `box[0, i]` internally.

## `_flush()`: the actual `InferApp` call

1. Builds the request, calls `InferApp.__call__()` — the sole inference entry point;
   `CLoPASession` never calls the raw session's `binary_predict()`/etc. directly.
2. Checks the returned affine matches what was sent (`torch.allclose`) — a corruption guard
   on the request/response round-trip.
3. Inverts the RAS reorientation applied at `set_image()` time via `reorient_from_ras()`,
   writing the result into `self._display_buffer` (a reference set independently in
   `set_target_buffer()`, *not* trusted to stay valid via `session.target_buffer` identity —
   `InferApp.load_new_image()` reallocates that on every `IS_interactive_init` request).
4. Resets the accumulator and flips `self._is_init = False` (so the next request is an
   `IS_interactive_edit`, not another full init).

## RAS orientation round-trip

`reorient_to_ras()` / `reorient_from_ras()` (module-level functions, not on the class) follow
the same pattern already established in `clopa/adaptation/io_operations.py`'s `WriteImage`
and IS_Validate's `write_image_utils/post.py`:

- **In**: seed `original_affine` on the `MetaTensor` meta dict, forward-`Orientation` to RAS.
- **Out**: deep-copy the reference (image) `MetaTensor`, substitute the prediction's array
  values into it, recover the original axcodes via `nib.orientations.aff2axcodes()` on the
  tracked `original_affine`, and reapply `Orientation` forward (targeting those original
  axcodes) with `trace_transform(False)` — **not** MONAI's `.inverse()` method. Dtype is cast
  back explicitly at the end (`result.numpy().astype(pred_np.dtype)`), matching
  `write_itk()`'s convention, since the transform pipeline itself doesn't preserve it — the
  duplicated reference's array otherwise inherits the *reference's* dtype, not the
  prediction's.
