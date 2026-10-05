# Capture protocol (one page)

Route 2: stock apps only. Follow each step exactly. Pick ONE tier.

## Before you start (all tiers)
- Turn on every light in the rooms. Open curtains in daytime.
- Clear the floor along walls if you can; close cupboard doors.
- Leave interior doors **fully open**.
- Do not walk through mirrors' field of view slowly; just keep moving (see "Avoid").

## Tier 3 — LiDAR (iPhone 12 Pro or newer **Pro** model)
1. Install **Stray Scanner** (free, App Store, by Stray Robots). Open it once and allow camera access.
2. Tap **Settings → Video** and make sure "Save depth" and "Save confidence" are ON (default).
3. Stand in the doorway of the first room. Hold the phone **upright (portrait), at chest height**.
4. Tap **Record**. Walk slowly (about one step per second) along the walls of each room,
   **1–2 m from the wall**, phone tilted ~30° down so the screen shows the floor-wall line.
5. In every room, do one slow full turn in the middle, then **tilt up once** to sweep the
   ceiling (2–3 seconds). Walk through each doorway **straight, not diagonally**.
6. Visit every room, then **return to where you started** and point at the same view as the
   first second of recording. Tap **Stop**. Typical: 1 minute per room.
7. Hand-off: Files app → *On My iPhone → Stray Scanner* → long-press the newest folder →
   **Compress** → AirDrop the `.zip` to the laptop.
8. Run: `scan2plan run lidar <unzipped folder>`

## Tier 2 — Video (any iPhone 15 or newer)
1. Camera app → **Video**, 1080p or 4K, **0.5×/1× lens = 1×** (do not zoom).
2. Same walk as LiDAR steps 3–6, but **slower: one step every 2 seconds**, and turn
   slowly (a full turn takes ~10 s). Keep something textured in view (furniture, door
   frames, sockets); never film a blank wall from closer than 1 m.
3. One clip for the whole property, ≤ 5 minutes. AirDrop the `.mov`.
4. Run: `scan2plan run video <clip.mov>`

## Tier 1 — Photos (any iPhone 15 or newer)
1. Camera app → **Photo**, **1×** lens, no zoom, no Live/Portrait mode, phone **upright**.
2. Per room take **6–8 photos**: stand in each corner and shoot diagonally across the room
   (4 photos), plus 1 photo of each doorway **from 1 m inside the room, showing the next room
   through it**. Each photo should share at least a third of its view with another photo.
3. Make one folder per room on the laptop (`kitchen/`, `bed1/`, `hall/` ...) and drop that
   room's photos in. Doorway photos go in the room you stood in.
4. Run: `scan2plan run photos <folder containing the room folders>`

## Avoid (all tiers)
- Pointing straight at mirrors, glass doors or windows at night for more than a second.
- Wet/glossy floors under direct light; dark rooms (turn lights on).
- People walking in front of the phone; fast pans; zooming.
- Starting or ending a recording facing a blank wall.

## Time and what you get
LiDAR: ~1 min/room capture, ~1–7 min processing. Video: ~1.5 min/room, ~30 min processing for
a 2 min clip. Photos: ~1 min/room, ~40 min for 40 photos (8 GB laptop; faster with 16 GB).
Output: `plan.json` + `plan.png` in `out/<name>_<tier>/` (e.g. `out/kitchen_photos/`).
