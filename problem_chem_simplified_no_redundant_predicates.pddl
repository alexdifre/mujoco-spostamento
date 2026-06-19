(define (problem chemistry-demo-01)
  (:domain chemistry-lab-mixing)

  (:objects
    home loc-in-1 loc-in-2 loc-in-3 loc-in-4 loc-in-5 loc-out-a loc-out-b - location
    in-1 in-2 in-3 in-4 in-5 out-1 out-2 - beaker
  )

  (:init
    ; Quantities are normalized: 1.0 means a full container, 0.0 means empty.
    (= (stage_cost) 0)
    (= (total_time) 0)
    (= (discrete_steps) 0)
    (= (w0 robot1) 0.5)
    (= (w1 robot1) 0.5)
    (= (duration-go-to-source) 1)
    (= (duration-go-to-output) 1)
    (= (duration-collect-liquid) 5)
    (= (duration-dispense) 5)
    (arm-at home)

    ; Initial pipette / end-effector pose in meters.
    (= (ee-x) -0.176735)
    (= (ee-y) 0.762445)
    (= (ee-z) 0.378191)

    ; Scattered table layout. Each symbolic location has an x, y, z pose.
    (= (loc-x home) -0.176735)
    (= (loc-y home) 0.762445)
    (= (loc-z home) 0.378191)

    (= (loc-x loc-in-1) -0.52)
    (= (loc-y loc-in-1) 0.58)
    (= (loc-z loc-in-1) 0.18)

    (= (loc-x loc-in-2) -0.30)
    (= (loc-y loc-in-2) 0.94)
    (= (loc-z loc-in-2) 0.20)

    (= (loc-x loc-in-3) -0.62)
    (= (loc-y loc-in-3) 0.82)
    (= (loc-z loc-in-3) 0.16)

    (= (loc-x loc-in-4) 0.05)
    (= (loc-y loc-in-4) 0.55)
    (= (loc-z loc-in-4) 0.19)

    (= (loc-x loc-in-5) -0.12)
    (= (loc-y loc-in-5) 1.02)
    (= (loc-z loc-in-5) 0.21)

    (= (loc-x loc-out-a) 0.48)
    (= (loc-y loc-out-a) 0.66)
    (= (loc-z loc-out-a) 0.18)

    (= (loc-x loc-out-b) 0.36)
    (= (loc-y loc-out-b) 0.96)
    (= (loc-z loc-out-b) 0.18)

    ; Bucket centers in meters. These x/y/z values are used as the
    ; end-effector pose for prelevare-liquid actions and to derive metrics.
    (= (bucket-x in-1) -0.52)
    (= (bucket-y in-1) 0.58)
    (= (bucket-z in-1) 0.18)

    (= (bucket-x in-2) -0.30)
    (= (bucket-y in-2) 0.94)
    (= (bucket-z in-2) 0.20)

    (= (bucket-x in-3) -0.62)
    (= (bucket-y in-3) 0.82)
    (= (bucket-z in-3) 0.16)

    (= (bucket-x in-4) 0.05)
    (= (bucket-y in-4) 0.55)
    (= (bucket-z in-4) 0.19)

    (= (bucket-x in-5) -0.12)
    (= (bucket-y in-5) 1.02)
    (= (bucket-z in-5) 0.21)

    (= (bucket-x out-1) 0.48)
    (= (bucket-y out-1) 0.66)
    (= (bucket-z out-1) 0.18)

    (= (bucket-x out-2) 0.36)
    (= (bucket-y out-2) 0.96)
    (= (bucket-z out-2) 0.18)

    (beaker-at in-1 loc-in-1)
    (beaker-at in-2 loc-in-2)
    (beaker-at in-3 loc-in-3)
    (beaker-at in-4 loc-in-4)
    (beaker-at in-5 loc-in-5)
    (beaker-at out-1 loc-out-a)
    (beaker-at out-2 loc-out-b)
    (output-beaker out-1)
    (output-beaker out-2)

    ; Five input beakers, with duplicate sources to let the planner trade off
    ; physical distance and safety cost.
    (source-liquid in-1 water)
    (source-liquid in-2 acid)
    (source-liquid in-3 base)
    (source-liquid in-4 water)
    (source-liquid in-5 acid)

    ; Movement time in seconds, precomputed from 3D Euclidean distance * 10.
    (= (time home home) 0)
    (= (time home loc-in-1) 0.350)
    (= (time home loc-in-2) 0.194)
    (= (time home loc-in-3) 0.402)
    (= (time home loc-in-4) 0.280)
    (= (time home loc-in-5) 0.237)
    (= (time home loc-out-a) 0.598)
    (= (time home loc-out-b) 0.515)

    (= (time loc-in-1 home) 0.350)
    (= (time loc-in-1 loc-in-1) 0)
    (= (time loc-in-1 loc-in-2) 0.380)
    (= (time loc-in-1 loc-in-3) 0.234)
    (= (time loc-in-1 loc-in-4) 0.513)
    (= (time loc-in-1 loc-in-5) 0.535)
    (= (time loc-in-1 loc-out-a) 0.902)
    (= (time loc-in-1 loc-out-b) 0.862)

    (= (time loc-in-2 home) 0.194)
    (= (time loc-in-2 loc-in-1) 0.380)
    (= (time loc-in-2 loc-in-2) 0)
    (= (time loc-in-2 loc-in-3) 0.307)
    (= (time loc-in-2 loc-in-4) 0.471)
    (= (time loc-in-2 loc-in-5) 0.177)
    (= (time loc-in-2 loc-out-a) 0.746)
    (= (time loc-in-2 loc-out-b) 0.594)

    (= (time loc-in-3 home) 0.402)
    (= (time loc-in-3 loc-in-1) 0.234)
    (= (time loc-in-3 loc-in-2) 0.307)
    (= (time loc-in-3 loc-in-3) 0)
    (= (time loc-in-3 loc-in-4) 0.650)
    (= (time loc-in-3 loc-in-5) 0.484)
    (= (time loc-in-3 loc-out-a) 1.000)
    (= (time loc-in-3 loc-out-b) 0.891)

    (= (time loc-in-4 home) 0.280)
    (= (time loc-in-4 loc-in-1) 0.513)
    (= (time loc-in-4 loc-in-2) 0.471)
    (= (time loc-in-4 loc-in-3) 0.650)
    (= (time loc-in-4 loc-in-4) 0)
    (= (time loc-in-4 loc-in-5) 0.450)
    (= (time loc-in-4 loc-out-a) 0.399)
    (= (time loc-in-4 loc-out-b) 0.462)

    (= (time loc-in-5 home) 0.237)
    (= (time loc-in-5 loc-in-1) 0.535)
    (= (time loc-in-5 loc-in-2) 0.177)
    (= (time loc-in-5 loc-in-3) 0.484)
    (= (time loc-in-5 loc-in-4) 0.450)
    (= (time loc-in-5 loc-in-5) 0)
    (= (time loc-in-5 loc-out-a) 0.629)
    (= (time loc-in-5 loc-out-b) 0.435)

    (= (time loc-out-a home) 0.598)
    (= (time loc-out-a loc-in-1) 0.902)
    (= (time loc-out-a loc-in-2) 0.746)
    (= (time loc-out-a loc-in-3) 1.000)
    (= (time loc-out-a loc-in-4) 0.399)
    (= (time loc-out-a loc-in-5) 0.629)
    (= (time loc-out-a loc-out-a) 0)
    (= (time loc-out-a loc-out-b) 0.291)

    (= (time loc-out-b home) 0.515)
    (= (time loc-out-b loc-in-1) 0.862)
    (= (time loc-out-b loc-in-2) 0.594)
    (= (time loc-out-b loc-in-3) 0.891)
    (= (time loc-out-b loc-in-4) 0.462)
    (= (time loc-out-b loc-in-5) 0.435)
    (= (time loc-out-b loc-out-a) 0.291)
    (= (time loc-out-b loc-out-b) 0)

    ; Safety penalty: coarse risk proxy from route length / crowded side crossing.
    (= (safety home home) 0)
    (= (safety home loc-in-1) 1.155)
    (= (safety home loc-in-2) 0.92)
    (= (safety home loc-in-3) 1.246)
    (= (safety home loc-in-4) 2.727)
    (= (safety home loc-in-5) 0.969)
    (= (safety home loc-out-a) 3.387)
    (= (safety home loc-out-b) 3.211)

    (= (safety loc-in-1 home) 1.155)
    (= (safety loc-in-1 loc-in-1) 0)
    (= (safety loc-in-1 loc-in-2) 1.134)
    (= (safety loc-in-1 loc-in-3) 0.891)
    (= (safety loc-in-1 loc-in-4) 3.142)
    (= (safety loc-in-1 loc-in-5) 1.393)
    (= (safety loc-in-1 loc-out-a) 0.4089)
    (= (safety loc-in-1 loc-out-b) 0.1024)

    (= (safety loc-in-2 home) 0.92)
    (= (safety loc-in-2 loc-in-1) 1.134)
    (= (safety loc-in-2 loc-in-2) 0)
    (= (safety loc-in-2 loc-in-3) 1.016)
    (= (safety loc-in-2 loc-in-4) 3.048)
    (= (safety loc-in-2 loc-in-5) 0.796)
    (= (safety loc-in-2 loc-out-a) 0.2628)
    (= (safety loc-in-2 loc-out-b) 0.4085)

    (= (safety loc-in-3 home) 1.246)
    (= (safety loc-in-3 loc-in-1) 0.891)
    (= (safety loc-in-3 loc-in-2) 1.016)
    (= (safety loc-in-3 loc-in-3) 0)
    (= (safety loc-in-3 loc-in-4) 3.446)
    (= (safety loc-in-3 loc-in-5) 1.311)
    (= (safety loc-in-3 loc-out-a) 0.3412)
    (= (safety loc-in-3 loc-out-b) 0.4821)

    (= (safety loc-in-4 home) 2.727)
    (= (safety loc-in-4 loc-in-1) 3.142)
    (= (safety loc-in-4 loc-in-2) 3.048)
    (= (safety loc-in-4 loc-in-3) 3.446)
    (= (safety loc-in-4 loc-in-4) 0)
    (= (safety loc-in-4 loc-in-5) 3)
    (= (safety loc-in-4 loc-out-a) 0.0780)
    (= (safety loc-in-4 loc-out-b) 0.0904)

    (= (safety loc-in-5 home) 0.969)
    (= (safety loc-in-5 loc-in-1) 1.393)
    (= (safety loc-in-5 loc-in-2) 0.796)
    (= (safety loc-in-5 loc-in-3) 1.311)
    (= (safety loc-in-5 loc-in-4) 3)
    (= (safety loc-in-5 loc-in-5) 0)
    (= (safety loc-in-5 loc-out-a) 0.1277)
    (= (safety loc-in-5 loc-out-b) 0.1268)

    (= (safety loc-out-a home) 3.387)
    (= (safety loc-out-a loc-in-1) 0.4089)
    (= (safety loc-out-a loc-in-2) 0.2628)
    (= (safety loc-out-a loc-in-3) 0.3412)
    (= (safety loc-out-a loc-in-4) 0.0780)
    (= (safety loc-out-a loc-in-5) 0.1277)
    (= (safety loc-out-a loc-out-a) 0)
    (= (safety loc-out-a loc-out-b) 0.985)

    (= (safety loc-out-b home) 3.211)
    (= (safety loc-out-b loc-in-1) 0.1024)
    (= (safety loc-out-b loc-in-2) 0.4085)
    (= (safety loc-out-b loc-in-3) 0.4821)
    (= (safety loc-out-b loc-in-4) 0.0904)
    (= (safety loc-out-b loc-in-5) 0.1268)
    (= (safety loc-out-b loc-out-a) 0.985)
    (= (safety loc-out-b loc-out-b) 0)

    (= (dose-size dose-1) 0.25)
    (= (dose-size dose-2) 0.5)
    (pipette-empty)

    (= (beaker-capacity out-1) 1)
    (= (beaker-capacity out-2) 1)

    ; Output-beaker state: each numeric fluent stores the remaining amount.
    (= (remaining-to-fill out-1 water) 0.5)
    (= (remaining-to-fill out-1 acid) 0.25)
    (= (remaining-to-fill out-1 base) 0.25)
    (= (remaining-to-fill out-2 water) 0.25)
    (= (remaining-to-fill out-2 acid) 0.25)
    (= (remaining-to-fill out-2 base) 0.5)

    ; Initial input containers: all five are full, each with one liquid.
    (= (qty in-1 water) 1)
    (= (qty in-1 acid) 0)
    (= (qty in-1 base) 0)

    (= (qty in-2 water) 0)
    (= (qty in-2 acid) 1)
    (= (qty in-2 base) 0)

    (= (qty in-3 water) 0)
    (= (qty in-3 acid) 0)
    (= (qty in-3 base) 1)

    (= (qty in-4 water) 1)
    (= (qty in-4 acid) 0)
    (= (qty in-4 base) 0)

    (= (qty in-5 water) 0)
    (= (qty in-5 acid) 1)
    (= (qty in-5 base) 0)

    ; Initial output containers: both are empty.
    (= (qty out-1 water) 0)
    (= (qty out-1 acid) 0)
    (= (qty out-1 base) 0)

    (= (qty out-2 water) 0)
    (= (qty out-2 acid) 0)
    (= (qty out-2 base) 0)
  )

  (:goal
    (and
      ; Goal: out-1 = 0.5 water + 0.25 acid + 0.25 base.
      (= (qty out-1 water) 0.5)
      (= (qty out-1 acid) 0.25)
      (= (qty out-1 base) 0.25)

      ; Goal: out-2 = 0.25 water + 0.25 acid + 0.5 base.
      (= (qty out-2 water) 0.25)
      (= (qty out-2 acid) 0.25)
      (= (qty out-2 base) 0.5)

      (pipette-empty)
      (= (remaining-to-fill out-1 water) 0)
      (= (remaining-to-fill out-1 acid) 0)
      (= (remaining-to-fill out-1 base) 0)
      (= (remaining-to-fill out-2 water) 0)
      (= (remaining-to-fill out-2 acid) 0)
      (= (remaining-to-fill out-2 base) 0)
    )
  )

  ; Equivalent to MinimizeExpressionOnFinalState:
  ; stage_cost already contains movement time, safety, and manipulation costs.
  ; Metric: stage_cost plus an explicit total-time penalty and a small step penalty.
  ; Note: since movement actions already include w0*time in stage_cost, this
  ; intentionally gives time an additional penalty through total_time.
  (:metric minimize
    (+ (stage_cost)
       (+ (* (total_time) (w0 robot1))
          (* 0.1 (discrete_steps)))))
)
