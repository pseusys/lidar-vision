# Draft email: data request to Plozza et al. (arXiv 2412.15000)

Fill the bracketed fields before sending. Addresses are not in the paper's abstract page; Michele Magno (senior author, ETH
Zurich Center for Project-Based Learning) is the natural first contact, with Davide Plozza (first author) in copy.

---

**Subject:** Request for the 2D-LiDAR recordings from your SAS 2024 person-tracker paper

Dear Dr. Magno and Mr. Plozza,

I am [NAME], [ROLE] at [AFFILIATION], working on person detection from knee-height 2D LiDAR. I read *Autonomous Navigation in
Dynamic Human Environments with an Embedded 2D LiDAR-based Person Tracker* (SAS 2024) with great interest — the DR-SPAAM and
Norfair pipeline at 20 Hz on a Jetson Xavier NX is the closest published system to what we are building, and your motion-capture
validation is a standard of ground truth that public 2D-LiDAR datasets do not reach.

I am writing to ask whether the three recordings described in Section [IV] — Stationary Robot, Moving Robot 1 and Moving Robot 2
— could be shared for research use, under whatever terms and licence you prefer.

**Why these recordings specifically.** We are developing a detector that carries an object-level memory *inside* the network, so
that evidence from earlier scans can raise a person's score before the detection threshold is applied, rather than after it in a
tracking stage. Our results so far are on FROG, which uses a Hokuyo UTM-30LX at the same 0.25° angular resolution as your
UTM-30LX-EW. Your recordings would let us test the model on a different robot, field of view and scan rate **without retraining**
— the cleanest check we could run that the approach is not fitted to one dataset. Your Vicon ground truth for both the people and
the robot would also let us separate detection error from odometry error, which our own data cannot do.

**What would be most useful**, in whatever form is convenient:

- raw scans with timestamps (the 1080-point, 270° scans at 20 Hz);
- robot odometry as recorded (50 Hz);
- the motion-capture tracks for the participants, and for the robot if they are separable;
- a note on the LiDAR's mounting height and orientation, and on how odometry and mocap were time-synchronised.

Any subset would still be valuable — even one of the three recordings would support the transfer test.

**What we would do with it.** Evaluation only: no redistribution, no training on your data, and results reported as a transfer
experiment with your paper cited as the source of the recordings and as the reference pipeline. We would be glad to share our
findings with you before publishing anything that uses them, and to acknowledge your group however you prefer.

If sharing the recordings is not possible, I would still welcome a short answer on two points: whether the mounting height of
45 cm was chosen for leg visibility, and whether you observed DR-SPAAM's detection quality limiting the tracker in the moving-robot
recordings. Both would be useful to us even without the data.

Thank you for your time, and for releasing such a clearly documented system.

Best regards,
[NAME]
[ROLE, AFFILIATION]
[EMAIL] · [PROJECT OR REPOSITORY LINK, IF YOU WANT TO SHARE IT]

---

## Notes for whoever sends this

- **Section number** in the request is a guess from the abstract; check the paper's own numbering before sending.
- **Do not promise co-authorship** on their behalf; the offer above is acknowledgement and early sight of results, which is ours
  to give.
- **The fallback question matters**: if they decline the data, the two questions still inform `TODO.md` (mounting height feeds
  the knee-height assumption; their view on detector-limited tracking is exactly the modularity gap this project argues).
- **Precondition on our side before using any of it:** temporal lags are currently counted in frames, so a 20 Hz sensor would
  silently double the coarse horizon from 0.8 s to 1.6 s. A49's "settings in physical units" must land first.
