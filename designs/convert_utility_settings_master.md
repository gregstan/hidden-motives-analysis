# Plan: Unify Utility Settings Conversion and Standardize Bitstring Formatting

## Context

The codebase has grown several overlapping mechanisms for representing, converting, and displaying utility settings (the 16-Boolean configuration dicts that define each utility function):

- `convert_utility_settings` (utilities.py) — handles dict↔tuple↔raw_str↔int conversions but cannot handle equation strings, legacy 14-key formats, or integer model indices.
- `_format_utility_bitstring` (utilities.py) — adds dashes (XXXX-XXXX-XXXX-XXXX) for human readability and Excel safety; called in 3 places but never exposed through `convert_utility_settings`.
- `model_key_maker` (analysis.py) — builds compact IC keys (`0000000000001011~equation`) using raw bitstrings; duplicates conversion logic.
- `create_file_name_suffix` (config.py) — produces file-name cache keys using a raw sorted bitstring; independently reimplements the same bit-concatenation.
- `equation_to_settings` (utilities.py) — a FUNCTION that maps equation strings back to settings; has a latent bug on line 1020 where the condition checks `isinstance(equation_to_settings, dict)` instead of `isinstance(equ_to_settings, dict)`, making the early-return cache path dead code.

The immediate driver for unification is the **14→16 setting expansion**: the IC JSON stores 476 models with 14-boolean tuple-string keys like `"(True, False, ..., True)"`, while the codebase now expects 16-key dicts. Code that reads those old keys may silently misindex or crash. `convert_utility_settings` has no way to translate them. Making it the single entry point for all conversion and translation — with explicit `input_settings_format` support — means this problem is handled once and is forward-compatible if settings are added again.

---

## Files to Modify

- [utilities.py](utilities.py) — main changes
- [analysis.py](analysis.py) — `model_key_maker` update + call-site audit
- [config.py](config.py) — `create_file_name_suffix` update

---

## Step 1 — Bug Fix: `equation_to_settings` variable shadowing (utilities.py ~line 1020)

**The bug:** Inside the `equation_to_settings` FUNCTION body, this line reads:
```python
if isinstance(equation_to_settings, dict) and all(isinstance(key, str) for key in equ_to_settings.keys()):
    return equ_to_settings
```
`equation_to_settings` refers to the outer FUNCTION itself — `isinstance(a_callable, dict)` is always False — so the early-return cache path is dead code; the function always recomputes.

**Fix:** Change `equation_to_settings` → `equ_to_settings` in that condition:
```python
if isinstance(equ_to_settings, dict) and all(isinstance(key, str) for key in equ_to_settings.keys()):
    return equ_to_settings
```

Zero behavioral change (just restores the intended caching skip).

---

## Step 2 — Expand `convert_utility_settings` (utilities.py ~line 1041)

### 2a. Signature change

```python
def convert_utility_settings(
    utility_settings: Union[
        Dict[str, bool],   # UtilitySettings dict (16 keys)
        Tuple[bool, ...],  # boolean tuple
        Tuple[int, ...],   # int (0/1) tuple
        str,               # bitstring, formatted bitstring, equation string, or JSON tuple-repr
        int,               # model index into all_utility_functions.csv
    ],
    into: Union[type, str] = str,   # CHANGED: default was tuple; see §2b
    template: Optional[Dict[str, bool]] = None,
    input_settings_format: Optional[Dict[str, bool]] = None,  # NEW: legacy schema map
    sort_alphabetically: bool = False,                         # NEW: sort vs canonical order
    file_paths: Optional[FilePaths] = None,                    # NEW: needed for equation/index lookup
    general_settings: Optional[GeneralSettings] = None,       # NEW: passed to equation_to_settings
) -> Union[Dict[str, bool], Tuple[bool, ...], Tuple[int, ...], str]
```

### 2b. `into` accepted values (expanded)

| `into` value | Output type | Description |
|---|---|---|
| `str` | `str` | **NEW default**: formatted bitstring `XXXX-XXXX-XXXX-XXXX` |
| `'raw_str'` | `str` | Raw 16-char bitstring `0101010101010101` (old default) |
| `'equation'` | `str` | Utility equation string (calls `build_utility_equation`) |
| `tuple` | `tuple[bool, ...]` | Boolean tuple (existing) |
| `dict` | `dict[str, bool]` | UtilitySettings dict (existing) |
| `int` | `tuple[int, ...]` | Tuple of 0/1 ints (existing, used by `model_key_maker`) |

**Note on default change:** `into` now defaults to `str` (formatted bitstring) instead of `tuple`. All existing call sites that pass `into=tuple` explicitly are unaffected. Call sites that relied on the default (`tuple`) need auditing — but `convert_utility_settings()` with no `into` argument is unusual; most calls are explicit.

### 2c. Input auto-detection (new code path before existing logic)

Detect the input type before the existing conversion logic:

```
if isinstance(utility_settings, int):
    → load row from all_utility_functions_dataframe() or generate_utility_settings(sort_by_k=True)[index]
    → convert resulting dict normally

elif isinstance(utility_settings, str):
    strip dashes if present → 16-char raw bitstring
    if looks like bitstring ("0"/"1" only, len 16):
        → convert to tuple of bools → proceed as tuple input
    elif looks like JSON tuple-repr ("(True, False, ...)"):
        → ast.literal_eval → tuple → proceed as tuple input
    elif looks like model_key_maker key ("XXXX~equation"):
        → strip after "~", parse bitstring part → proceed
    else:
        → treat as equation string; call equation_to_settings(equation_function, ...) to get dict
```

### 2d. `input_settings_format` parameter — legacy 14→16 key translation

When `input_settings_format` is provided, it is a dict with the OLD schema keys (e.g., 14 keys). The function:
1. Builds a key map from `input_settings_format` keys to canonical `config.utility_settings` keys.
2. Starts with the current full 16-key schema, all False.
3. Fills in values from the input using the mapped keys.
4. Missing keys (the 2 new ones: `include_welfare_efficiency_term`, `include_relative_income_penalty`) default to False.

This means `convert_utility_settings(old_14_bool_tuple, input_settings_format=old_14_key_dict)` produces a correct 16-key dict silently.

The function should also accept the 14-element JSON tuple-repr string WITHOUT `input_settings_format` if and only if the tuple length is 14 — in that case, it tries to guess the format by matching positionally against the first 14 keys of the canonical schema (with a warning). If `input_settings_format` is provided, the mapping is explicit and no guessing occurs.

### 2e. `sort_alphabetically` parameter

When converting dict → tuple/str/int:
- `sort_alphabetically=True`: sort keys alphabetically (current behavior in `create_file_name_suffix`)
- `sort_alphabetically=False` (default): use insertion order; optionally validate the dict's key order matches canonical `config.utility_settings` order (emit a warning if not)

---

## Step 3 — Delete `_format_utility_bitstring` and update its call sites

`_format_utility_bitstring` is a private helper that does exactly what `convert_utility_settings(..., into=str)` will now do. Delete the function (utilities.py ~line 2688–2701) and replace all 3 call sites:

| Old call | Replacement |
|---|---|
| `utilities.py:1622` — `_format_utility_bitstring(raw_bitstring=raw_utility_bitstring)` | `convert_utility_settings(raw_utility_bitstring, into=str)` |
| `utilities.py:1873` — `_format_utility_bitstring(raw_bitstring=raw)` | `convert_utility_settings(raw, into=str)` |
| `utilities.py:2037` — `_format_utility_bitstring(raw_bitstring=raw_bit)` | `convert_utility_settings(raw_bit, into=str)` |

The XXXX-XXXX-XXXX-XXXX formatting logic (`f"{r[0:4]}-{r[4:8]}-{r[8:12]}-{r[12:16]}"`) moves inline into the `into=str` dispatch block inside `convert_utility_settings`.

---

## Step 4 — Update `model_key_maker` (analysis.py ~line 1293)

Current key format: `"0000000000001011~Uᵢ(A)=..."` (raw 16-char bitstring)  
New key format: `"0000-0000-0000-1011~Uᵢ(A)=..."` (formatted bitstring)

**Change the `into=str` branch:**
```python
# OLD:
return str(model)[1:-1].replace(", ", "") + "~" + build_utility_equation(...)
# NEW (let convert_utility_settings format it):
return convert_utility_settings(model, into=str) + "~" + build_utility_equation(...)
```

**Update the `into=tuple` reverse-conversion:**
```python
# OLD: model.split("~")[0].isdigit()
# NEW: strip dashes before checking isdigit
bit_part = model.split("~")[0].replace("-", "")
if isinstance(model, str) and bit_part.isdigit():
    return tuple(int(dig) for dig in bit_part)
```

**Write a migration helper** (add to utilities.py):
```python
def migrate_model_keys_in_ic_json(ic_json_path: str) -> None:
    """
    One-time migration: converts raw-bitstring model keys to formatted XXXX-XXXX-XXXX-XXXX~eq keys.
    Creates a backup at ic_json_path + '.bak' before writing.
    Safe to re-run (already-formatted keys are left unchanged).
    """
```

---

## Step 5 — Update `create_file_name_suffix` (config.py ~line 518)

**Change the final for loop** from raw concatenation to formatted bitstring via `convert_utility_settings`:

```python
# OLD:
file_name_suffix += "-"
for key, val in sorted(utility_settings.items()):
    file_name_suffix += f"{int(val)}"

# NEW (use double-dash as section separator to avoid ambiguity with bitstring dashes):
sorted_us = dict(sorted(utility_settings.items()))
file_name_suffix += "--" + convert_utility_settings(sorted_us, into=str, sort_alphabetically=True)
```

This changes the suffix from `~11010101-0101010101010101` to `~11010101--0101-0101-0101-0101` (double-dash cleanly separates general settings from the XXXX-XXXX-XXXX-XXXX utility block).

**Migration note:** Existing cached files have the old suffix format. Add a companion helper:
```python
def migrate_file_name_suffix(directory: str, dry_run: bool = True) -> list[str]:
    """
    Renames cached files whose suffix uses the old single-dash format to the new double-dash format.
    With dry_run=True, returns the list of planned renames without executing them.
    """
```

---

## Step 6 — Call-Site Audit for `into=str` Change

`convert_utility_settings(..., into=str)` currently returns a raw 16-char string. After Step 2, it will return `XXXX-XXXX-XXXX-XXXX`. Every call site must be checked:

**Call sites that need `'raw_str'` (still need unformatted):**
- Any place that indexes into the bitstring by position (e.g., `bits[3]`) — must switch to `into='raw_str'`
- Any place that splits the string character-by-character — must switch to `into='raw_str'`
- `_format_utility_bitstring(raw_bitstring=convert_utility_settings(..., into=str))` pattern — would double-format; switch inner call to `into='raw_str'`

**Call sites where formatted output is fine:**
- Display/print contexts
- Column values in DataFrames shown to users
- Any place that just checks length == 16 (update check to 19)

**Likely call sites to audit** (from exploration):
- `utilities.py:1621` — used as input to `_format_utility_bitstring` → switch to `into='raw_str'`
- `utilities.py:1873` — same pattern → switch to `into='raw_str'`
- `utilities.py:2035-2036` — check usage
- `analysis.py:1302` — `into=int`, unaffected
- `analysis.py:1707` — `str(convert_utility_settings(..., into=tuple))` — uses `into=tuple`, unaffected
- `behavioral_distances.py:171` — `into=dict`, unaffected

---

## Step 7 — Fix Stale 14-Key Assertion (analysis.py ~line 4908)

```python
# OLD (raises on 16-key dicts):
assert len(list(child_utility_settings[0].keys())) == 14

# NEW:
assert len(list(child_utility_settings[0].keys())) == len(config.utility_settings)
```

Also update any docstring comments that say "All 14 Boolean keys must be present" to say 16 (search: `utilities.py:1288`, `utilities.py:1466`, `behavioral_distances.py:1186`).

---

## Implementation Order

1. **Step 1** — bug fix (5 min, zero risk)
2. **Step 7** — fix 14-key assertion and update "14" comments (10 min, zero risk)
3. **Step 2a–2b** — expand `convert_utility_settings` signature and `into` dispatch; inline XXXX-XXXX-XXXX-XXXX formatting logic for `into=str`
4. **Step 2c** — add input auto-detection paths (additive)
5. **Step 2d** — add `input_settings_format` legacy translation (additive)
6. **Step 2e** — add `sort_alphabetically` parameter (additive)
7. **Step 3** — delete `_format_utility_bitstring`; update its 3 call sites to use `convert_utility_settings(..., into=str)`
8. **Step 6** — audit all remaining `into=str` call sites; switch raw-string users to `into='raw_str'`
9. **Step 4** — update `model_key_maker` key format + write IC JSON migration helper
10. **Step 5** — update `create_file_name_suffix` + write file-rename migration helper
11. **Compile check** — `python -m py_compile utilities.py analysis.py config.py`
12. **Smoke test** — inline test at bottom of utilities.py (can be deleted after)

---

## Verification

```python
import config
from utilities import convert_utility_settings, _format_utility_bitstring

us = config.utility_settings  # 16-key dict

# Roundtrip: dict → formatted str → raw → dict
fmt = convert_utility_settings(us, into=str)
assert len(fmt) == 19 and fmt[4] == '-'  # XXXX-XXXX-XXXX-XXXX

raw = convert_utility_settings(us, into='raw_str')
assert len(raw) == 16 and '-' not in raw

back = convert_utility_settings(raw, into=dict)
assert back == us

# Legacy 14-key input
old_format_keys = {k: False for k in list(us.keys())[:14]}
translated = convert_utility_settings(tuple([False]*14), input_settings_format=old_format_keys, into=dict)
assert len(translated) == 16
assert translated['include_welfare_efficiency_term'] == False

# Integer index
us_from_idx = convert_utility_settings(0, into=dict)
assert len(us_from_idx) == 16

# Equation roundtrip
eq = convert_utility_settings(us, into='equation')
assert isinstance(eq, str) and 'U' in eq

# model_key_maker backward compat
from analysis import model_key_maker
key = model_key_maker(us, into=str)
assert key[4] == '-'                          # formatted bitstring
back_tuple = model_key_maker(key, into=tuple)
assert len(back_tuple) == 16

# create_file_name_suffix format
from config import create_file_name_suffix, general_settings
suffix = create_file_name_suffix(general_settings, us)
assert '--' in suffix                         # double-dash separates sections
```

Also run `python -m py_compile utilities.py analysis.py config.py` with zero errors.

Confirm `_format_utility_bitstring` no longer exists anywhere in the codebase:
```
grep -r "_format_utility_bitstring" *.py   # should return nothing
```
