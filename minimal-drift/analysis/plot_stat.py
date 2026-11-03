from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


stat_file = Path("minimal-drift/minimal-drift.stat")

columns = [
    "t", "s", "numParticles", "charge", "energy",
    "rms_x", "rms_y", "rms_s",
    "rms_px", "rms_py", "rms_ps",
    "emit_x", "emit_y", "emit_s",
    "mean_x", "mean_y", "mean_s",
    "ref_x", "ref_y", "ref_z",
    "ref_px", "ref_py", "ref_pz",
    "max_x", "max_y", "max_s",
    "xpx", "ypy", "zpz",
    "Dx", "DDx", "Dy", "DDy",
    "Bx_ref", "By_ref", "Bz_ref",
    "Ex_ref", "Ey_ref", "Ez_ref",
    "dE", "dt", "partsOutside",
    "DebyeLength", "plasmaParameter", "temperature",
    "rmsDensity", "nBins",
]

lines = stat_file.read_text().splitlines()

data_marker = lines.index("&data")

data_end = next(
    i for i in range(data_marker + 1, len(lines))
    if lines[i].strip() == "&end"
)

# After &end come the parameter values:
# processors, revision, species.
numeric_lines = lines[data_end + 4:]

rows = []

for line in numeric_lines:
    values = line.split()

    if len(values) != len(columns):
        continue

    try:
        rows.append([float(value) for value in values])
    except ValueError:
        continue

data = np.asarray(rows)

col = {name: i for i, name in enumerate(columns)}

s = data[:, col["s"]]
rms_x_mm = data[:, col["rms_x"]] * 1e3
rms_y_mm = data[:, col["rms_y"]] * 1e3

plt.figure()
plt.plot(s, rms_x_mm, label=r"$\sigma_x$")
plt.plot(s, rms_y_mm, label=r"$\sigma_y$")

plt.xlabel("Path length s [m]")
plt.ylabel("RMS beam size [mm]")
plt.title("Minimal drift: transverse beam size")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("analysis/minimal-drift-rms.png", dpi=200)