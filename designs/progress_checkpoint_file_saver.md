# Plan: ProgressCheckpoint — General-Purpose Long-Run Checkpointing

"""
Purpose: this document is a general-purpose implementation template for the ProgressCheckpoint
class and the two-file staging protocol it enforces.  It is designed to apply across
repositories.  The final section records project-specific adoption for the Hidden Motives repo.
Save this plan to the plans/ folder once it is finalized (per AGENTS.md §9 convention).
Use the descriptive filename plans/progress_checkpoint_protocol.md.
"""

---

## The Problem

Long-running computations that accumulate results incrementally are vulnerable to total data
loss if the process is killed.  The fix is to flush completed work to a staging file as the
loop progresses, then rename it to the final file only when computation is fully done.

Success criteria:
  • A process killed mid-run can be restarted and resumes with zero re-computation of
    already-completed outer-loop iterations.
  • Downstream functions that read the output file never see a partial result.
  • Parallelism (mp.Pool / imap_unordered) works without file locking.
  • Adopting the protocol in a new function takes ~8 lines of wrapper code.

---

## Two-File Staging Protocol

```
<name>_unfinished.<ext>    written incrementally; signals computation is in progress
<name>.<ext>               written once, on completion; downstream functions only read this
<name>_unfinished_.<ext>   fallback backup when _unfinished is locked (open in Excel, etc.)
```

Downstream code treats any existing final file as a complete result.
An _unfinished file means computation was interrupted; the next run resumes from it.
The backup (_unfinished_) is created when the primary _unfinished file is locked.
On the next startup the backup is merged into the primary, then deleted.

---

## What ProgressCheckpoint IS

`ProgressCheckpoint` is a Python **class**.  Calling `ProgressCheckpoint(...)` returns
an **instance** — an object that holds all state for one long-running computation:
the accumulated rows or dict, the set of completed outer-loop items, file paths,
timing information, and configuration.  The instance is created once at the top of the
long-running function and lives for the duration of that function call.

The class does NOT subclass anything.  It is a self-contained utility class.

---

## Class Specification

### Constructor

```python
class ProgressCheckpoint:
    """
    Manages incremental disk saves for long-running computations that build up a list of
    records (CSV mode) or a nested dict (JSON mode).

    Two-file staging protocol:
      _unfinished file   — written incrementally during computation.
      Final file         — written once, atomically, on completion; never partial.
      _unfinished_ file  — Windows fallback when the _unfinished file is locked.

    Usage pattern:
        checkpoint: ProgressCheckpoint = ProgressCheckpoint(
            file_path=file_paths['processed'],
            file_name='my_results.csv',
            outer_loop_items=all_items,
            completed_key_cols=['item_id'],
            create_new_file=create_new_file,
        )
        if (existing := checkpoint.load_if_complete()) is not None:
            return existing
        for item in checkpoint.pending_items():
            rows: list[dict] = compute(item)   # rows must contain item_id column
            checkpoint.save(rows=rows)
        return checkpoint.finalize()

    Arguments:
        • file_path: str
            Directory path for the output file.  If file_name is None, file_path is
            treated as the full path including the filename.
        • file_name: str | None
            Filename (e.g. 'results.csv').  Joined with file_path internally.
            Pass None when file_path already includes the filename.
        • outer_loop_items: Sequence
            The complete sequence of items the outer loop iterates over, known upfront.
            Used by pending_items() to skip already-completed items on resume.
            Items must be hashable (strings, ints, tuples).
        • completed_key_cols: list[str] | None
            CSV only.  The column name(s) in saved rows that identify which outer-loop
            item produced those rows (e.g. ['player_uuid']).  Multiple rows may share
            the same key value — all rows for one outer iteration share one key.
            Used on resume to determine which items from outer_loop_items are already done
            by reading the _unfinished file.
            If None, completion is tracked by count: the first N items in outer_loop_items
            are assumed done where N is the number of completed saves recorded.
        • file_format: str | None
            'csv' or 'json'.  Inferred from file_name extension if None.
        • save_every_n_outer_iters: int
            Flush accumulated data to the _unfinished file after this many outer iterations.
            Default 1.  Increase only when each outer iteration takes < 30 seconds.
        • create_new_file: bool
            If True, delete any existing _unfinished and final files before starting,
            so computation restarts from scratch.  Pass the function's own create_new_file
            argument here — the class handles all early-return and cleanup logic.
        • verbose: bool
            Print resume and checkpoint-save messages (default True).
        • n_write_retries: int
            Number of retry attempts on PermissionError or OSError before falling back to
            the backup path (default 4; ~4 seconds of total wait at 1s intervals).
            Covers both 'file open in another app' and 'previous process still releasing
            the lock' — identical failure mode on Windows.
        • retry_delay_seconds: float
            Seconds between write retries (default 1.0).
    """
    def __init__(
        self,
        file_path: str,
        file_name: str | None = None,
        outer_loop_items: Sequence | None = None,
        completed_key_cols: list[str] | None = None,
        file_format: str | None = None,
        save_every_n_outer_iters: int = 1,
        create_new_file: bool = False,
        verbose: bool = True,
        n_write_retries: int = 4,
        retry_delay_seconds: float = 1.0,
    ) -> None: ...
```

### Path derivation (inside __init__)

```python
final_path: str      = os.path.join(file_path, file_name) if file_name else file_path
base: str, ext: str  = os.path.splitext(final_path)
unfinished_path: str = f"{base}_unfinished{ext}"
backup_path: str     = f"{base}_unfinished_{ext}"
```

### create_new_file handling (inside __init__, before loading checkpoint)

```python
if create_new_file:
    for path_to_clear in (unfinished_path, backup_path, final_path):
        if os.path.exists(path_to_clear):
            os.remove(path_to_clear)
```

### Checkpoint loading (inside __init__, after create_new_file handling)

```
1.  If backup_path exists: merge its rows/keys into unfinished_path, delete backup.
2.  If unfinished_path exists: load rows (CSV) or dict (JSON).
      CSV: extract unique values of completed_key_cols — these are the already-done items.
      JSON: extract top-level dict keys — these are the already-done items.
    Store them in self._completed_outer_keys (set).
3.  If verbose and _completed_outer_keys is non-empty:
      print 'Resuming from checkpoint: N items already complete out of M total.'
```

### Public API

```python
def load_if_complete(self) -> pd.DataFrame | dict | None:
    """
    If create_new_file was False and the final output file already exists, load and
    return it.  Otherwise return None.  Always call this immediately after construction.
    """

def pending_items(self) -> Generator:
    """
    Yield each item from outer_loop_items that is not yet in the completed set.
    The class tracks which item was most recently yielded so that save() can record
    completion without the caller needing to pass the item key explicitly.

    Prints a skip message for each already-complete item when verbose=True.
    """

def save(self, rows: list[dict]) -> None:
    """
    Accumulate rows from the most recently yielded pending item and flush to the
    _unfinished file every save_every_n_outer_iters calls.

    The rows must contain the columns named in completed_key_cols so the checkpoint
    can verify the correct item is being saved and — on a future resume — identify
    which items are already done.

    After save() returns, the current item is added to completed_keys() and will be
    skipped on the next run if the process is interrupted before finalize().

    Prints on flush: 'Checkpoint saved: K/N items  elapsed Xm Ys'
    """

def finalize(self) -> pd.DataFrame | dict:
    """
    Flush any remaining accumulated data, write the final output file, delete the
    _unfinished file, and return the result.
    Raises ValueError if zero rows were accumulated — never writes an empty final file.
    """

def completed_keys(self) -> set:
    """
    Return the set of outer-loop item keys already saved in the _unfinished checkpoint.
    Populated on construction from any existing checkpoint file and extended by save().
    """
```

### Private methods

```python
def _try_write(self, path: str, write_fn: Callable[[str], None]) -> bool:
    """
    Attempt write_fn(path) up to n_write_retries times, waiting retry_delay_seconds
    between each.  Returns True on success, False after all retries fail.
    """

def _save_checkpoint(self) -> None:
    """
    Serialize accumulated data to unfinished_path via _try_write.
    If all retries fail, fall back to backup_path and print a notification.
    """

def _merge_backup(self) -> None:
    """
    Load backup_path into memory, merge with existing unfinished_path data
    (checkpoint rows take priority to avoid duplicates), overwrite unfinished_path,
    delete backup_path.
    """
```

### CSV vs JSON mode summary

| Aspect | CSV mode | JSON mode |
|---|---|---|
| Accumulator type | `list[dict]` | `dict` |
| _save_checkpoint writes | `pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')` | `json.dump(d, fh, ensure_ascii=False, indent=2)` |
| _merge_backup strategy | concat DataFrames, drop_duplicates on completed_key_cols | `{**backup_dict, **checkpoint_dict}` (checkpoint wins) |
| completed_keys() source | unique tuple(s) from completed_key_cols in loaded DataFrame | `set(accumulated_dict.keys())` |
| finalize() writes | DataFrame → final CSV | dict → final JSON |

---

## Annotated Demo (sequential, CSV mode)

```python
import os
import pandas as pd
from typing import Generator
from preprocessing import ProgressCheckpoint


def compute_slow_analysis(
    items: list[str],
    general_settings: dict,
    file_paths: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
    """
    Illustrates the standard ProgressCheckpoint pattern for a sequential row-accumulating
    function.  This demo can be adapted to any function that fits that shape.
    """

    "Construct a ProgressCheckpoint instance — one instance per long-running function call."
    """
    On construction the class:
      1. Joins file_path + file_name to derive the final, _unfinished, and _backup paths.
      2. If create_new_file=True, deletes any existing _unfinished and final files.
      3. If an _unfinished file exists from a previous interrupted run, loads it.
         completed_keys() then returns the item_id values already saved in that file.
      4. Prints a resume message if any completed items were found.
    """
    checkpoint: ProgressCheckpoint = ProgressCheckpoint(
        file_path=file_paths['processed'],
        file_name='slow_analysis_results.csv',
        outer_loop_items=items,           # full list, known upfront; hashable items
        completed_key_cols=['item_id'],   # rows must contain an 'item_id' column
        create_new_file=create_new_file,
    )

    """
    load_if_complete() returns the final DataFrame if the final file already exists and
    create_new_file=False.  Returns None when computation needs to proceed.
    """
    existing_result: pd.DataFrame | None = checkpoint.load_if_complete()
    if existing_result is not None:
        return existing_result

    """
    pending_items() yields only the items from outer_loop_items that are NOT yet in
    completed_keys().  On the very first run, that is all items.  On a resumed run,
    already-complete items are silently (or verbosely) skipped.

    The class tracks which item was most recently yielded so that save() can mark
    the right item as complete without the caller passing the key explicitly.
    """
    for item_id in checkpoint.pending_items():

        "Slow computation — can take seconds to hours per item."
        result_rows: list[dict] = [
            {
                'item_id':   item_id,   # must be present; matches completed_key_cols
                'metric_a':  compute_a(item_id),
                'metric_b':  compute_b(item_id),
            }
        ]

        """
        save() accumulates result_rows, extracts item_id from the rows to mark this
        item as done in completed_keys(), and flushes to the _unfinished file every
        save_every_n_outer_iters calls.  If the process is killed immediately after
        save() returns, this item will be skipped on the next run.
        """
        checkpoint.save(rows=result_rows)

    """
    finalize() flushes any remaining rows, writes the final CSV, deletes the _unfinished
    file, and returns the complete DataFrame.  Raises ValueError if 0 rows were accumulated
    (never writes an empty output file that would crash downstream readers).
    """
    return checkpoint.finalize()
```

### Multiple rows per outer iteration (the common case)

If each outer iteration produces many rows (e.g. 5 folds × 29 models per player), all those
rows share the same completed_key_cols value:

```python
for player_uuid in checkpoint.pending_items():
    player_rows: list[dict] = []
    for fold_id in range(n_folds):
        for model_idx in candidate_models:
            player_rows.append({
                'player_uuid': player_uuid,  # same key across all rows for this player
                'fold_id':     fold_id,
                'model_idx':   model_idx,
                ...
            })
    checkpoint.save(rows=player_rows)        # all 145 rows saved together; one outer key
```

On resume, the checkpoint reads the _unfinished file, finds all unique `player_uuid` values,
and sets those as the completed keys — so `pending_items()` skips them automatically.

---

## Parallel Pattern (mp.Pool / imap_unordered)

Pool workers never call save() — they return data to the master.  The master calls save()
sequentially as results arrive via imap_unordered.  No file locking needed.

```python
def compute_slow_parallel(
    items: list[str],
    file_paths: dict,
    create_new_file: bool = False,
) -> pd.DataFrame:
    checkpoint: ProgressCheckpoint = ProgressCheckpoint(
        file_path=file_paths['processed'],
        file_name='slow_parallel_results.csv',
        outer_loop_items=items,
        completed_key_cols=['item_id'],
        create_new_file=create_new_file,
    )
    if (existing := checkpoint.load_if_complete()) is not None:
        return existing

    n_workers: int = max(os.cpu_count() - 1, 1)
    with mp.Pool(processes=n_workers) as pool:
        """
        imap_unordered yields (item_id, rows) tuples one at a time to the master process.
        The master calls checkpoint.save() sequentially — no concurrent writes.
        """
        for item_id, result_rows in pool.imap_unordered(_worker_fn, checkpoint.pending_items()):
            checkpoint.save(rows=result_rows)

    return checkpoint.finalize()
```

Note: when passing `checkpoint.pending_items()` directly to imap_unordered, the generator
is consumed by the pool.  The class must handle `save()` being called in a different order
than items were yielded (imap_unordered returns results out of order).  Implementation must
use the `item_id` in the returned rows (via completed_key_cols) to mark completion rather
than relying on yield order.  One implementation strategy: `pending_items()` yields items
and also registers them as "in-flight"; `save()` marks the item from the rows as complete
regardless of which item was most recently yielded.

---

## What the class does NOT cover

| Pattern | Why excluded |
|---|---|
| AMPD distance matrix | Matrix-filling with NaN-based completion; bespoke logic; mature; leave as-is |
| IC analysis | Per-config JSON warm-start baked into the analysis; leave as-is |
| typological_model_comparison_fit_population | Already saves every N subsets; leave as-is |

---

## Project-Specific Adoption (Hidden Motives repo)

**First and only adoption at this time**: `compute_cross_validated_architecture_losses`.
Retrofit other functions later using this template after the first adoption is verified.

### Files to modify

| File | Change |
|---|---|
| `preprocessing.py` | Add `ProgressCheckpoint` class after line 119 with section header |
| `analysis.py` | Refactor `compute_cross_validated_architecture_losses` to use checkpoint |
| `AGENTS.md` | Add "Long-running computation checkpointing" section |
| `plans/progress_checkpoint_protocol.md` | Save a copy of this plan (per AGENTS.md §9) |

### compute_cross_validated_architecture_losses adoption sketch

```python
"Replace the existing early-return cache-check block:"
checkpoint: ProgressCheckpoint = ProgressCheckpoint(
    file_path=file_paths['processed'],
    file_name='cv_architecture_losses.csv',
    outer_loop_items=all_player_uuids,
    completed_key_cols=['player_uuid'],
    create_new_file=create_new_file,
)
if (existing_df := checkpoint.load_if_complete()) is not None:
    print(f"Loaded from cache: {output_csv_path}  ({len(existing_df)} rows)")
    return existing_df

"Replace the main loop header:"
for player_uuid_key in checkpoint.pending_items():
    participant_rows: list[dict] = []
    "... inner fold × model loops unchanged, appending to participant_rows ..."
    checkpoint.save(rows=participant_rows)

"Replace the final to_csv + empty-guard:"
cv_losses_df: pd.DataFrame = checkpoint.finalize()
print(f"Saved: {output_csv_path}  ({len(cv_losses_df)} rows)")
return cv_losses_df
```

### AGENTS.md update

Add a "Long-running computation checkpointing" section after "Progress printing" and before
"Inner helper functions" covering:
  1. When to use ProgressCheckpoint (> ~30 minutes, row-accumulating or dict-accumulating output).
  2. The two-file protocol semantics (_unfinished vs. final).
  3. Sequential and parallel (master-collects) patterns in one paragraph each.
  4. Do not retrofit AMPD matrix or IC analysis.
  5. Reference plans/progress_checkpoint_protocol.md for the full template.

---

## Verification

1. Delete cv_architecture_losses.csv and cv_architecture_losses_unfinished.csv if present.
2. Run main.py; kill after 3–4 participants complete.  Confirm _unfinished.csv exists with those rows.
3. Re-run: confirm skip messages appear for already-done participants; computation continues.
4. Run to completion: confirm final CSV exists; _unfinished file is gone;
   compute_h_form_cross_validated loads it correctly.
5. Run with create_new_file=True: confirm _unfinished is deleted and computation restarts.
6. While step 3 is running, open _unfinished.csv in Excel: confirm backup-fallback message
   and _unfinished_.csv creation; close Excel; re-run and confirm backup merges on startup.
