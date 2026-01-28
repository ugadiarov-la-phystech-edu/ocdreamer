import collections
import os
import re
import uuid

import numpy as np


class CometOutput:
  def __init__(self, name, config, fps, pattern=r'.*'):
    self._pattern = re.compile(pattern)
    self._fps = fps
    import comet_ml

    run_id = os.getenv('COMET_RUN_ID', None)
    mode = 'create'
    if run_id is not None:
      mode = 'get'

    experiment = comet_ml.start(experiment_key=run_id, mode=mode)
    experiment.log_parameters(dict(config))
    experiment.set_name(os.getenv('COMET_EXPERIMENT_NAME', name))
    self._experiment = experiment

  def __call__(self, summaries):
    bystep = collections.defaultdict(dict)
    experiment = self._experiment
    for step, name, value in summaries:
      if not self._pattern.search(name):
        continue
      if isinstance(value, str):
          # Comet is not happy with logging of strings
          continue
      elif len(value.shape) == 0:
        bystep[step][name] = float(value)
      elif len(value.shape) == 1:
        experiment.log_histogram_3d(value, name=name, step=step)
      elif len(value.shape) in (2, 3):
        value = value[..., None] if len(value.shape) == 2 else value
        if value.shape[-1]==17:
          continue
        assert value.shape[-1] in [1, 3, 4], value.shape
        if value.dtype != np.uint8:
          value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
        value = np.transpose(value, [2, 0, 1])
        experiment.log_image(value, name=name, step=step)
      elif len(value.shape) == 4:
        if value.shape[-1]==17:
          continue
        from moviepy.editor import ImageSequenceClip
        # Sanity check that the channel dimension is the last
        assert value.shape[-1] in [1, 3, 4], f"Invalid shape: {value.shape}"
        if value.shape[-1] == 1:
            value = np.repeat(value, 3, axis=-1)
        elif value.shape[-1] == 4:
            value = value[..., :3]
        # If the video is a float, convert it to uint8
        if np.issubdtype(value.dtype, np.floating):
          value = np.clip(255 * value, 0, 255).astype(np.uint8)
        value = list(value)
        clip = ImageSequenceClip(value, fps=self._fps)
        path = f'/tmp/{uuid.uuid4()}.mp4'
        need_remove = False
        try:
          clip.write_videofile(path)
          need_remove = True
          experiment.log_video(path, name=name, step=step)
        finally:
          if need_remove:
            os.remove(path)

    for step, metrics in bystep.items():
      metrics['global_step'] = step
      self._experiment.log_metrics(metrics, step=step)
