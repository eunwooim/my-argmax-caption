# ESPER-style analysis: before RL vs after RL

### COCO val (in-distribution)

| metric | before RL | after RL |
|---|---|---|
| words / caption | 90.8 | 57.0 |
| sentences / caption | 5.33 | 3.47 |
| BLEU-4 vs human (ESPER Tab. 1) | 0.039 | 0.064 |
| CIDEr-D vs human (ESPER Tab. 1) | 0.000 | 0.008 |
| CHAIR-i: share of mentioned objects that are wrong | 0.166 | 0.110 |
| CHAIR-s: captions with any wrong object | 0.328 | 0.206 |
| repeated 3-gram rate | 0.009 | 0.004 |
| distinct-1 | 0.079 | 0.097 |
| distinct-2 | 0.361 | 0.401 |
| vocabulary size | 3579 | 2770 |
| share of unique captions | 1.000 | 1.000 |
| proper nouns / caption | 1.14 | 0.80 |
| count words / caption | 0.53 | 0.43 |

Words used more after RL: barriers (0→4), hitting (0→3), intricate (0→3), toward (0→3), skiers (0→3), enter (1→5), rusted (1→5), shore (2→7), present (1→4), narrow (1→4), sniffing (1→4), away (2→6), drinks (2→6), riders (3→8), posing (4→10)

Words used less after RL: conveys (36→0), sense (28→0), daytime (26→0), captures (20→0), typical (16→0), day (61→3), suggests (88→6), nostalgic (11→0), adding (44→3), mood (10→0), architecture (9→0), sunny (46→4), atmosphere (122→13), overall (176→20), habitat (15→1)

Most common openings before RL: "the image shows…" 84%, "this image shows…" 5%, "this image captures…" 2%, "the image depicts…" 2%, "this image depicts…" 2%

Most common openings after RL: "the image shows…" 94%, "the image is…" 3%, "the image depicts…" 2%, "this image shows…" 0%, "the image features…" 0%

### DOCCI test (out-of-distribution)

| metric | before RL | after RL |
|---|---|---|
| words / caption | 92.0 | 63.4 |
| sentences / caption | 5.31 | 3.88 |
| BLEU-4 vs human (ESPER Tab. 1) | 0.046 | 0.030 |
| CIDEr-D vs human (ESPER Tab. 1) | 0.116 | 0.055 |
| CHAIR-i: share of mentioned objects that are wrong | 0.368 | 0.296 |
| CHAIR-s: captions with any wrong object | 0.325 | 0.205 |
| repeated 3-gram rate | 0.010 | 0.006 |
| distinct-1 | 0.119 | 0.145 |
| distinct-2 | 0.458 | 0.505 |
| vocabulary size | 2189 | 1842 |
| share of unique captions | 1.000 | 1.000 |
| proper nouns / caption | 1.14 | 0.76 |
| count words / caption | 0.41 | 0.38 |

Words used more after RL: all (0→3), root (0→3), packed (0→3), line (0→3), land (1→5), dots (1→4), system (1→4), owls (1→4), steps (1→4), densely (1→3), fixtures (1→3), directional (1→3), resembling (1→3), morning (1→3), together (1→3)

Words used less after RL: daytime (11→0), sense (10→0), gives (7→0), cozy (6→0), indoor (6→0), maintained (6→0), typical (12→1), outdoor (31→4), planter (5→0), emblem (5→0), clean (5→0), environment (37→6), suggests (54→10), scenic (4→0), visually (4→0)

Most common openings before RL: "the image shows…" 81%, "the image depicts…" 16%, "this image captures…" 2%, "the image appears…" 0%, "this image is…" 0%

Most common openings after RL: "the image shows…" 92%, "the image depicts…" 6%, "the image appears…" 0%, "the image is…" 0%, "the image features…" 0%

Examples: gallery.html
