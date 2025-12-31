{
  VibeML - A tiny neural network in Pascal
  Inspired by Karpathy's micrograd: just the math, nothing else.
}
program nn;
uses Math;

var
  w1: array[0..1, 0..3] of Double;  { input->hidden weights }
  w2: array[0..3] of Double;         { hidden->output weights }
  b1: array[0..3] of Double;         { hidden biases }
  b2: Double;                        { output bias }
  h: array[0..3] of Double;          { hidden activations }
  o, do_, y: Double;
  dh: array[0..3] of Double;
  X: array[0..3, 0..1] of Double = ((0,0), (0,1), (1,0), (1,1));
  Y: array[0..3] of Double = (0, 1, 1, 0);
  epoch, s, i, j: Integer;

function sigmoid(x: Double): Double;
begin sigmoid := 1 / (1 + exp(-x)); end;

begin
  Randomize;
  
  { Init weights }
  for i := 0 to 1 do for j := 0 to 3 do w1[i,j] := Random * 2 - 1;
  for j := 0 to 3 do begin w2[j] := Random * 2 - 1; b1[j] := Random * 2 - 1; end;
  b2 := Random * 2 - 1;

  { Train }
  for epoch := 1 to 10000 do
    for s := 0 to 3 do begin
      { Forward }
      for j := 0 to 3 do h[j] := sigmoid(X[s,0]*w1[0,j] + X[s,1]*w1[1,j] + b1[j]);
      o := sigmoid(h[0]*w2[0] + h[1]*w2[1] + h[2]*w2[2] + h[3]*w2[3] + b2);
      
      { Backward }
      do_ := (o - Y[s]) * o * (1 - o);
      for j := 0 to 3 do dh[j] := do_ * w2[j] * h[j] * (1 - h[j]);
      
      { Update }
      for j := 0 to 3 do begin
        w2[j] := w2[j] - 0.5 * do_ * h[j];
        b1[j] := b1[j] - 0.5 * dh[j];
        w1[0,j] := w1[0,j] - 0.5 * dh[j] * X[s,0];
        w1[1,j] := w1[1,j] - 0.5 * dh[j] * X[s,1];
      end;
      b2 := b2 - 0.5 * do_;
    end;

  { Test }
  WriteLn('XOR Neural Network');
  for s := 0 to 3 do begin
    for j := 0 to 3 do h[j] := sigmoid(X[s,0]*w1[0,j] + X[s,1]*w1[1,j] + b1[j]);
    o := sigmoid(h[0]*w2[0] + h[1]*w2[1] + h[2]*w2[2] + h[3]*w2[3] + b2);
    WriteLn(X[s,0]:1:0, ' XOR ', X[s,1]:1:0, ' = ', o:0:4, ' (expected ', Y[s]:1:0, ')');
  end;
end.

