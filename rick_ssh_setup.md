# SSH into Rick's Machine — Setup Log & Continuation Guide

> Written 2026-05-16. Assumes complete retrograde amnesia on both sides.

---

## Why we're doing this

The information criterion (IC) analysis in the hidden motives paper compares **476 utility functions** using AIC and BIC to find the best-fitting utility representation for the UBM. The problem is that ideally this analysis should use the **full dynamic Bayesian UBM** — meaning it fits individual-level belief-updating parameters across all **73 participants** for each of the 476 utility functions. That's an enormous computation.

On Greg's machine (6 cores), the full dynamic version would take **months**. Rick's machine has **28 cores**. That's enough to make it feasible.

The manuscript currently has a footnote in the IC section acknowledging this limitation (static, no-updating approximation used for computational tractability). Running the full version on Rick's machine eliminates that limitation and strengthens the paper.

### The one-line code change

In [main.py](main.py) around line 204, the IC analysis is called like this:

```python
information_criterion_analysis(general_settings=general_settings, utility_settings=utility_settings,
    file_paths=file_paths, param_bds=param_bds, max_iters=24, robustness_epsilon=36, check_for_n_players='all')
```

To enable full dynamic updating, add `dynamic_updating=True`:

```python
information_criterion_analysis(general_settings=general_settings, utility_settings=utility_settings,
    file_paths=file_paths, param_bds=param_bds, max_iters=24, robustness_epsilon=36, check_for_n_players='all',
    dynamic_updating=True)
```

Also make sure `run_code_settings['run_information_criterion_analysis']` is set to `True` in main.py (line 12) before running.

What `dynamic_updating=True` does internally (see [analysis.py](analysis.py) line 1616): it sets `update_method = 'grid'` and `general_settings['use_particle_filter'] = True`, switching the predictor fitting from a static approximation to full Bayesian belief updating across observed game histories.

---

## Rick's machine

| Property | Value |
|---|---|
| SSH address | `l-w6ffp4xmmx.psych.lsa.umich.edu` |
| Username | `gregstan` |
| OS | macOS |
| Logical CPU cores | 28 |
| Python version | 3.14.4 |

---

## VPN requirement

Rick's machine is only reachable from off-campus via the **UMich VPN**. This is free — do not buy a commercial VPN.

1. Go to `its.umich.edu` and search "VPN"
2. Download and install **Cisco Secure Client** for Windows
3. Open it, connect to: `vpngate.umich.edu`
4. Log in with your UMich uniqname + password + Duo 2FA

If you're on the UMich campus network you can skip the VPN.

---

## SSH setup status (already completed 2026-05-16)

The following has already been done — you do not need to redo it:

- [x] Greg's password on Rick's machine has been changed from the temporary one
- [x] Greg's SSH public key has been added to `~/.ssh/authorized_keys` on Rick's machine
- [x] Passwordless login confirmed working

Greg's SSH public key (for reference):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBSFQt44PkszF2BUAHyXZiEg4Q0h3Fk7y+/fPCzJgldp gregory stanley@DESKTOP-9U75192
```

---

## How to connect

Open any terminal (Git Bash, Windows Terminal, PowerShell) and run:

```bash
ssh gregstan@l-w6ffp4xmmx.psych.lsa.umich.edu
```

No password prompt means everything is working correctly. If it asks for a password, something changed — see the troubleshooting note at the bottom.

---

## Next steps (not yet done)

### 1. Get the code onto Rick's machine

Make sure Greg's latest changes are pushed to GitHub first:

```bash
git push
```

Then on Rick's machine (via SSH):

```bash
git clone https://github.com/gregstan/hidden-motives-analysis
cd hidden-motives-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify the environment:

```bash
python3 -c "import numpy; print('good')"
```

### 2. Make the code change

In `main.py`:
- Set `run_code_settings['run_information_criterion_analysis'] = True`
- Add `dynamic_updating=True` to the `information_criterion_analysis(...)` call (see above)

Push those changes from Greg's machine, then on Rick's machine:

```bash
git pull
```

### 3. Run the analysis

Still on Rick's machine, with the venv active:

```bash
python3 main.py
```

The analysis writes results incrementally, so if the run is interrupted it can be resumed (the `write_mode="resume"` default handles this).

### 4. Get the results back

Results are written to the `bic_aic/` directory. Once the run finishes, you can either:
- `git push` from Rick's machine (if you set up git credentials there), or
- Use `scp` from Greg's machine to pull the output files:

```bash
scp -r gregstan@l-w6ffp4xmmx.psych.lsa.umich.edu:~/hidden-motives-analysis/bic_aic/ ./bic_aic_from_rick/
```

---

## Troubleshooting

**SSH asks for a password when it shouldn't:**
The key in `~/.ssh/authorized_keys` on Rick's machine may have been lost. SSH in with your password, then re-run the key setup:
```bash
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBSFQt44PkszF2BUAHyXZiEg4Q0h3Fk7y+/fPCzJgldp gregory stanley@DESKTOP-9U75192" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Connection times out:**
You're probably off-campus and not on the UMich VPN. Connect to `vpngate.umich.edu` first.

**`nproc` not found on Rick's machine:**
It's a Mac — use `sysctl -n hw.logicalcpu` instead. Returns 28.
