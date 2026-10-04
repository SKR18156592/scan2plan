"""Rendered floor plan: rooms, walls with dimensions, openings."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = plt.cm.Pastel1.colors


def render_plan(result, path, title=None, show_sigma=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, room in enumerate(result["rooms"]):
        P = np.array(room["polygon"])
        ax.fill(P[:, 0], P[:, 1], color=COLORS[i % len(COLORS)], zorder=1)
        for w in room["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            ax.plot(*zip(a, b), color="k" if w["observed"] else "0.6", lw=3, solid_capstyle="round", zorder=3,
                    ls="-" if w["observed"] else "--")
            d = b - a
            L = np.linalg.norm(d)
            for op in w["openings"]:
                u = d / max(L, 1e-9)
                p0, p1 = a + u * op["offset"], a + u * (op["offset"] + op["width"]["value"])
                col = {"door": "tab:red", "opening": "tab:orange", "window": "tab:blue"}[op["type"]]
                ax.plot(*zip(p0, p1), color=col, lw=6, zorder=4)
                m = (p0 + p1) / 2
                ax.annotate(f"{op['type']} {op['width']['value']*100:.0f}", m, fontsize=6, color=col, ha="center", zorder=6)
            if L >= 0.4:
                m = (a + b) / 2
                n = np.array([-d[1], d[0]]) / max(L, 1e-9)
                inside = m + n * 0.12
                txt = f"{L:.2f}"
                if show_sigma:
                    txt += f"\n±{w['length']['ci95'][1]-L:.2f}"
                ax.annotate(txt, inside, fontsize=6, ha="center", va="center", zorder=5,
                            rotation=0 if abs(d[0]) > abs(d[1]) else 90)
        c = P.mean(0)
        ch = room["ceiling_height"]
        lines = [room["name"], f"{room['floor_area']['value']:.1f} m²"]
        if ch["value"] is not None:
            lines.append(f"h {ch['value']:.2f} m")
        ax.annotate("\n".join(lines), c, ha="center", va="center", fontsize=8, weight="bold", zorder=6)
    for e in result.get("adjacency", []):
        pass
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or result.get("capture", ""), fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
