;; Problem definition
(define (problem problem-1)

  ;; Specifying the domain for the problem
  (:domain student-life-domain)

  ;; Objects definition
  (:objects
    ; Buildings
    B1
    B2
    B3
    B4
    ; People (Students and teachers)
    S1
    T1
    T2
    ; Lectures
    L1
    L2
    L3
  )

  ;; Intial state of problem 1
  (:init
    ;; Declaration of the objects
    ; Buildings
    (BUILDING B1)
    (BUILDING B2)
    (BUILDING B3)
    (BUILDING B4)
    ; People (Students and teachers)
    (PERSON S1)
    (PERSON T1)
    (PERSON T2)
    ; Lectures
    (LECTURE L1)
    (LECTURE L2)
    (LECTURE L3)
        
    ;; Declaration of the predicates of the objects
    ; We set people locations
    (is-person-at S1 B4)
    (is-person-at T1 B2)
    (is-person-at T2 B3)
   
    ; We set the buildings where lectures take place
    (IS-LECTURE-AT L1 B1)
    (IS-LECTURE-AT L2 B2)
    (IS-LECTURE-AT L3 B3)
    (IS-LECTURE-AT L1 B4)
    (IS-LECTURE-AT L3 B4)
    
    ; We set whether the lecture is a morning or afternoon lecture (or both)
    (IS-MORNING L1)
    (IS-MORNING L2)
    (IS-AFTERNOON L2)
    (IS-AFTERNOON L3)

    ; We set whether the building has a restuarant
    (HAS-RESTURANT B3)

    ; We set wheter the person is a teacher or a student
    (IS-TEACHER T1)
    (IS-TEACHER T2)
    (IS-STUDENT S1)

    ; We set which teacher teaches which lecture
    (teaches-lecture T1 L1)
    (teaches-lecture T1 L2)
    (teaches-lecture T2 L3)
    
    ; We set the connections between the buildings
    (IS-CONNECTED B1 B2) (IS-CONNECTED B2 B1) 
    (IS-CONNECTED B2 B3) (IS-CONNECTED B3 B2) 
    (IS-CONNECTED B3 B4) (IS-CONNECTED B4 B3) 
  )

  ;; Goal specification
  (:goal
    (and
      (attended-lecture S1 L1)
      (attended-lecture S1 L2)
      (attended-lecture S1 L3)

      (had-lunch S1)
    )
  )

)
