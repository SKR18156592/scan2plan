"""Concealed-damage rules and scope line items.

Concealed damage is inferred, never observed, so each flag names the rule
that fired and the visible evidence it fired on. Rules are deliberately few
and conservative (common restoration heuristics):

CD-01 ceiling moisture: water stain or mould on a ceiling -> possible leak
      in the floor/roof void above.
CD-02 wall-base moisture: water stain, mould or peeling paint whose lower
      edge is within 0.3 m of the floor -> possible plumbing leak or rising
      damp behind the wall.
CD-03 opening-corner crack: crack touching the corner of a door/window
      opening -> possible structural movement / failed lintel.
CD-04 extensive mould: mould region over 0.5 m^2 -> likely growth behind the
      lining (visible mould is usually the smaller part).
"""
import numpy as np

MOISTURE = {"water_stain", "mold", "peeling_paint"}


def concealed_flags(regions):
    flags = []
    for k, r in enumerate(regions):
        rid = r["id"]
        if r["surface_type"] == "ceiling" and r["cls"] in ("water_stain", "mold"):
            flags.append(dict(rule="CD-01", region=rid, surface=r["surface"],
                              finding="possible leak in floor/roof void above ceiling"))
        if r["surface_type"] == "wall" and r["cls"] in MOISTURE and r.get("height") is not None:
            lower_edge = r["height"] - r["extent"][1] / 2
            if lower_edge < 0.3:
                flags.append(dict(rule="CD-02", region=rid, surface=r["surface"],
                                  finding="possible plumbing leak or rising damp behind wall"))
        if r["cls"] == "crack" and r.get("near_opening"):
            flags.append(dict(rule="CD-03", region=rid, surface=r["surface"],
                              finding="possible structural movement at opening corner"))
        if r["cls"] == "mold" and r["area"]["value"] > 0.5:
            flags.append(dict(rule="CD-04", region=rid, surface=r["surface"],
                              finding="mould likely extends behind lining"))
    return flags


def scope_items(regions, plan):
    """One or more line items per damaged surface. Quantities are metric with
    the interval of the measurement they derive from."""
    walls = {w["id"]: (w, r) for r in plan["rooms"] for w in r["walls"]}
    rooms = {r["id"]: r for r in plan["rooms"]}
    items = []

    def wall_area(w, room):
        h = room["ceiling_height"]["value"] or 2.4
        hs = room["ceiling_height"]["sigma"] or 0.15
        L, Ls = w["length"]["value"], w["length"]["sigma"]
        return L * h, float(np.hypot(Ls * h, L * hs))

    for r in regions:
        s = r["surface"]
        if r["surface_type"] == "wall" and s in walls:
            w, room = walls[s]
            A, As = wall_area(w, room)
            if r["cls"] == "crack":
                items.append(dict(surface=s, region=r["id"], item="Rake out and fill crack, sand, spot prime",
                                  unit="m", quantity=_q(max(r["extent"]), 0.25 * max(r["extent"]))))
            elif r["cls"] == "hole":
                items.append(dict(surface=s, region=r["id"], item="Patch drywall (cut back to sound board, patch, tape, skim)",
                                  unit="m2", quantity=_q(r["area"]["value"] + 0.1, r["area"]["sigma"])))
            elif r["cls"] == "mold":
                items.append(dict(surface=s, region=r["id"], item="Mould treatment incl. 0.5 m margin",
                                  unit="m2", quantity=_q(_margin_area(r, 0.5), r["area"]["sigma"] * 2)))
            # Every wall defect ends in repainting the whole wall face.
            items.append(dict(surface=s, region=r["id"],
                              item="Stain-block prime and repaint wall face" if r["cls"] in MOISTURE else "Repaint wall face",
                              unit="m2", quantity=_q(A, As)))
        elif r["surface_type"] in ("ceiling", "floor"):
            room = rooms[r["room"]]
            A, As = room["floor_area"]["value"], room["floor_area"]["sigma"]
            if r["surface_type"] == "ceiling":
                if r["cls"] in ("crack", "hole"):
                    items.append(dict(surface=s, region=r["id"], item="Repair ceiling plaster/board",
                                      unit="m2", quantity=_q(_margin_area(r, 0.3), r["area"]["sigma"] * 2)))
                items.append(dict(surface=s, region=r["id"], item="Stain-block prime and repaint ceiling",
                                  unit="m2", quantity=_q(A, As)))
            else:
                items.append(dict(surface=s, region=r["id"], item=f"Inspect/repair floor finish ({r['cls']})",
                                  unit="m2", quantity=_q(_margin_area(r, 0.3), r["area"]["sigma"] * 2)))
    return items


def _margin_area(r, m):
    a, b = r["extent"]
    return (a + 2 * m) * (b + 2 * m)


def _q(v, s):
    return dict(value=round(float(v), 3), sigma=round(float(s), 3),
                ci95=[round(float(v - 1.96 * s), 3), round(float(v + 1.96 * s), 3)])
