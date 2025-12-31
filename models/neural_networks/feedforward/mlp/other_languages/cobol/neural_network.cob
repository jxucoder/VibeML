*> VibeML - A tiny neural network in COBOL
*> Inspired by Karpathy's micrograd: just the math, nothing else.

IDENTIFICATION DIVISION.
PROGRAM-ID. NN.

DATA DIVISION.
WORKING-STORAGE SECTION.
01 W1.  05 W1R OCCURS 2.  10 W1V OCCURS 4 PIC S9V9(6).
01 W2.  05 W2V OCCURS 4 PIC S9V9(6).
01 B1.  05 B1V OCCURS 4 PIC S9V9(6).
01 B2   PIC S9V9(6).
01 H.   05 HV OCCURS 4 PIC S9V9(6).
01 O    PIC S9V9(6).
01 DO   PIC S9V9(6).
01 DH.  05 DHV OCCURS 4 PIC S9V9(6).
01 XV.  05 X1 PIC S9V9. 05 X2 PIC S9V9.
01 YV   PIC S9V9.
01 TMP  PIC S9(3)V9(6).
01 I    PIC 9. 01 J PIC 9. 01 S PIC 9. 01 E PIC 9(5).

PROCEDURE DIVISION.
    *> Init weights randomly
    PERFORM VARYING I FROM 1 BY 1 UNTIL I > 2
      PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
        COMPUTE W1V(I,J) = (FUNCTION RANDOM - 0.5) * 2
      END-PERFORM
    END-PERFORM
    PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
      COMPUTE W2V(J) = (FUNCTION RANDOM - 0.5) * 2
      COMPUTE B1V(J) = (FUNCTION RANDOM - 0.5) * 2
    END-PERFORM
    COMPUTE B2 = (FUNCTION RANDOM - 0.5) * 2

    *> Train 10000 epochs
    PERFORM VARYING E FROM 1 BY 1 UNTIL E > 10000
      PERFORM VARYING S FROM 0 BY 1 UNTIL S > 3
        *> Set input/target for sample S
        IF S = 0 MOVE 0 TO X1 X2 YV END-IF
        IF S = 1 MOVE 0 TO X1 MOVE 1 TO X2 YV END-IF
        IF S = 2 MOVE 1 TO X1 MOVE 0 TO X2 MOVE 1 TO YV END-IF
        IF S = 3 MOVE 1 TO X1 X2 MOVE 0 TO YV END-IF
        
        *> Forward: hidden layer
        PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
          COMPUTE TMP = X1*W1V(1,J) + X2*W1V(2,J) + B1V(J)
          COMPUTE HV(J) = 1 / (1 + FUNCTION EXP(0 - TMP))
        END-PERFORM
        
        *> Forward: output
        COMPUTE TMP = HV(1)*W2V(1) + HV(2)*W2V(2) 
                    + HV(3)*W2V(3) + HV(4)*W2V(4) + B2
        COMPUTE O = 1 / (1 + FUNCTION EXP(0 - TMP))
        
        *> Backward
        COMPUTE DO = (O - YV) * O * (1 - O)
        PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
          COMPUTE DHV(J) = DO * W2V(J) * HV(J) * (1 - HV(J))
        END-PERFORM
        
        *> Update weights
        PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
          COMPUTE W2V(J) = W2V(J) - 0.5 * DO * HV(J)
          COMPUTE B1V(J) = B1V(J) - 0.5 * DHV(J)
          COMPUTE W1V(1,J) = W1V(1,J) - 0.5 * DHV(J) * X1
          COMPUTE W1V(2,J) = W1V(2,J) - 0.5 * DHV(J) * X2
        END-PERFORM
        COMPUTE B2 = B2 - 0.5 * DO
      END-PERFORM
    END-PERFORM

    *> Test
    DISPLAY "XOR Neural Network"
    PERFORM VARYING S FROM 0 BY 1 UNTIL S > 3
      IF S = 0 MOVE 0 TO X1 X2 YV END-IF
      IF S = 1 MOVE 0 TO X1 MOVE 1 TO X2 YV END-IF
      IF S = 2 MOVE 1 TO X1 MOVE 0 TO X2 MOVE 1 TO YV END-IF
      IF S = 3 MOVE 1 TO X1 X2 MOVE 0 TO YV END-IF
      PERFORM VARYING J FROM 1 BY 1 UNTIL J > 4
        COMPUTE TMP = X1*W1V(1,J) + X2*W1V(2,J) + B1V(J)
        COMPUTE HV(J) = 1 / (1 + FUNCTION EXP(0 - TMP))
      END-PERFORM
      COMPUTE TMP = HV(1)*W2V(1) + HV(2)*W2V(2) 
                  + HV(3)*W2V(3) + HV(4)*W2V(4) + B2
      COMPUTE O = 1 / (1 + FUNCTION EXP(0 - TMP))
      DISPLAY X1 " XOR " X2 " = " O " (expected " YV ")"
    END-PERFORM
    STOP RUN.

