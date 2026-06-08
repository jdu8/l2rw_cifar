"""HDF5 pair storage.

Each L2RW step that is sampled for logging is written as a group:
  /epoch_{E}/batch_{B}/
      train_embeddings  [N_train, D]  float32
      train_losses      [N_train]     float32
      val_embeddings    [N_val, D]    float32
      val_losses        [N_val]       float32
      l2rw_weights      [N_train]     float32
  attributes: epoch, batch_idx

After every write the HDF5 file is flushed and synced to disk so that
partial runs are not lost on crash.
"""
import os

import h5py
import numpy as np
import torch


class PairStorage:
    def __init__(self, output_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._path = output_path
        # Open in append mode so we can resume from a crash
        self._file = h5py.File(output_path, "a")

    # ------------------------------------------------------------------
    def write(
        self,
        *,
        epoch: int,
        batch_idx: int,
        train_embeddings: torch.Tensor,
        train_losses: torch.Tensor,
        val_embeddings: torch.Tensor,
        val_losses: torch.Tensor,
        l2rw_weights: torch.Tensor,
    ) -> None:
        """Write one L2RW step's data and immediately sync to disk."""
        grp_key = f"epoch_{epoch}/batch_{batch_idx}"
        # Skip if already written (e.g. resuming from crash mid-epoch)
        if grp_key in self._file:
            return

        grp = self._file.require_group(grp_key)
        grp.attrs["epoch"] = epoch
        grp.attrs["batch_idx"] = batch_idx

        def _np(t: torch.Tensor) -> np.ndarray:
            return t.detach().float().cpu().numpy()

        grp.create_dataset("train_embeddings", data=_np(train_embeddings), compression="gzip")
        grp.create_dataset("train_losses",     data=_np(train_losses),     compression="gzip")
        grp.create_dataset("val_embeddings",   data=_np(val_embeddings),   compression="gzip")
        grp.create_dataset("val_losses",       data=_np(val_losses),       compression="gzip")
        grp.create_dataset("l2rw_weights",     data=_np(l2rw_weights),     compression="gzip")

        # Flush HDF5 library buffers, then fsync the underlying file
        self._file.flush()
        try:
            fd = self._file.id.get_vfd_handle()
            os.fsync(fd)
        except Exception:
            # Some VFDs don't expose a file descriptor; flush is still done
            pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._file.id.valid:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
