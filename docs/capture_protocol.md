# Capture protocol (one page): stock apps only

Follow every step literally. Each tier has its own app and walk. You hand over one folder or file per capture.

## Before you start (all tiers)
- Turn on every light and open interior doors fully. Leave curtains as they are.
- Put a sticky note on any mirror or glass door at eye height (any colour). Mirrors and glass show up as fake rooms without it.
- Phone: any iPhone 15 or newer. Close other apps and charge to at least 30%.
- Hold the phone upright at chest height. Move slowly: about one step per second, and turn no faster than a slow head turn.

## Tier 3, LiDAR (iPhone Pro / Pro Max only): app "Stray Scanner" (free, App Store)
1. Install **Stray Scanner** (Stray Robots). The defaults are fine; no settings to change.
2. Start at the front door and tap **Record**.
3. In each room: walk the perimeter about 1 m from the walls, aiming the phone at the wall in front of you. Do one full lap, then aim at the ceiling for about 3 s and at the floor for about 3 s.
4. Go through every doorway slowly and aim straight at it for about 1 s. Every opening is measured from these views.
5. **Finish where you started** (the front door) and tap Stop. Returning to the start gives the drift correction a loop closure.
6. Expect 30–60 s per room. Avoid pointing at a mirror for more than a second.
7. Hand-off: Stray Scanner → share → Files/AirDrop the scan folder (or zip) to the laptop. Then run:
   `scan2plan run <folder-or-zip>`

## Tier 2, video (any iPhone 15+): native Camera app
1. Camera app → **Video**, **1x** lens, 1080p or 4K at 30 fps. Hold the phone **landscape**.
2. Walk the same route as above, but stay about 1.5 m from the walls and aim **across** each room so that two walls are always in view. Never aim at a blank wall from up close.
3. In each room, also turn in place once slowly (about 10 s per 360°).
4. Go through doorways slowly, aiming through them for about 2 s. End at the start point.
5. Hand-off: AirDrop the .MOV to the laptop, then run `scan2plan run <video.MOV> --tier video`

## Tier 1, photos (any iPhone 15+): native Camera app, 2–8 photos per room
1. Camera app → **Photo**, **1x**, landscape.
2. For each room, stand in a corner (in the doorway if the room is small). Take 4–6 photos while turning in place, each overlapping about 1/3 with the previous one, until you have covered the whole room. Make sure the floor–wall line is visible in each photo.
3. If the room is L-shaped or longer than 5 m, take 2 more from the opposite corner.
4. **Every photo of a doorway must show both door jambs.**
5. Hand-off: make one folder per room, numbered in walking order (`01_hall`, `02_kitchen`, ...) inside a property folder. Run:
   `scan2plan run <property_folder> --tier photo`

## What to avoid (all tiers)
- Running, fast turns, or covering the camera/LiDAR with fingers.
- People or pets walking through the shot.
- Very low light: if the screen looks grainy, turn on more lights.
- Mirrors, glass, and wet/glossy floors: mark them with a sticky note and don't film them for long.

## Ground truth (benchmark only): laser measurer
Fill `benchmark/ground_truth_template.csv`, one row per measurement:
- Each wall: wall-to-wall at about 1 m height along the wall, in cm to 1 decimal.
- Each opening: clear width between jambs at about 1 m height.
- Ceiling height: floor to ceiling at the room centre.
- Photograph the laser display for each reading; the photos are part of the raw data.
