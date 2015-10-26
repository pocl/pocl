(* -*-Mathematica-*- script to determine the polynomial coefficients
   to approximate functions
   
   2013-02-12 Erik Schnetter <eschnetter@perimeterinstitute.ca>
   
   Based on the "minimax" algorithm, but using least squares instead *)

funcs =
  {
    {Sin[2 Pi #]&,           0, 1/4,       1, 2, 10,   1, 1, 1,   {0, 1/4}},
    {Cos[2 Pi #]&,           0, 1/4,       0, 2, 10,   1, 1, 1,   {0, 1/4}},
 (* {Tan[Pi #]&,             0, 1/2,       1, 2, 3,    2, 2, 3,   {0, 1/4}}, *)
    {Log[2, (1+#)/(1-#)]&, -3 + 2 Sqrt[2], 3 - 2 Sqrt[2],
                                           0, 1, 15,   0, 1, 0,   {0}},
    {2^#&,                   -1/2, +1/2,   0, 1, 15,   0, 1, 1,   {}},
    {ArcTan,                 0, 1/2,       1, 2, 15,   1, 2, 1,   {0, 1}}
  };

findcoeffs[func_, xmin_, xmax_, dmin_, dstep_, ndegrees_, denndegrees_,
           conslocs_] :=
  Module[{prec, npts,
          degree,
          qs, q2x, x2q,
          funcq,
          funcqpade, polydenom,
          coeffs, powers, poly,
          norm,
          A, r, b,
          approx,
          error1, error2, error3, error},
         
         Print[{"function ", func, ndegrees, denndegrees}];
         
         (* Working precision *)
         prec = 30;
         
         (* Number of test points *)
         npts = 10000;
         
         degree = dmin + ndegrees dstep;
         
         (* Test points in normalized interval [0,1] *)
         qs = Table[n/(npts-1), {n, 0, npts-1}];
         
         (* A (discrete) L2-norm based on the test points *)
         norm[f_] = Sqrt[Total[N[(Abs[f[#1]]^2 &) /@ qs, prec]] / Length[qs]];
         
         (* Transform q to x coordinate *)
         q2x[q_] = xmin + (xmax - xmin) q;
         x2q[x_] = (x - xmin) / (xmax - xmin);
         
         (* Function with rescaled input *)
         funcq[q_] = func[q2x[q]];
         
         (* Use denominator of Pade approximant as denominator of our
            approximant *)
         funcqpade[q_] = PadeApproximant[funcq[q], {q, 0, denndegrees}];
         polydenom[q_] = Denominator[funcqpade[q]];
         
         (* List of expansion coefficients *)
         coeffs = Table[c[i], {i, dmin, degree, dstep}];
         
         (* Corresponding list of powers of q *)
         powers[q_] = Table[If[i==0, 1, q^i], {i, dmin, degree, dstep}];
         
         (* Polynomial *)
         poly[q_] = coeffs . powers[q];
         
         (* We determine the expansion coefficients via a
            least-squares method *)
         A = N[Table[powers[q], {q, qs}], prec];
         r = N[(funcq[#1] polydenom[#1] &) /@ qs, prec];
         (* r = N[(Limit[funcq[q] polydenom[q], q->#1] &) /@ qs, prec]; *)
         b = LeastSquares[A, r];
         
         (* Define approximating polynomial using this solution *)
         approx[q_] = ((poly[q] /. MapThread[#1 -> #2 &, {coeffs, b}]) /
                       N[polydenom[q], prec]);
         
         (* Calculate three kinds of errors to check solution: *)
         (* (1) the (discrete) norm form above: *)
         error1 = norm[approx[#1] - funcq[#1] &];
         (* (2) A non-discrete L2-norm using an integral: *)
         error2 = Sqrt[NIntegrate[Abs[approx[q] - funcq[q]]^2, {q, 0, 1},
                                  WorkingPrecision -> prec]];
         (* (3) The maximum of the error: *)
         error3 = NMaxValue[{Abs[approx[q] - funcq[q]], q >= 0 && q <= 1}, {q},
                            WorkingPrecision -> prec];
         error = Max[error1, error2, error3];
         
         (* Evaluate at constraint locations to check solution: *)
         consvals = Map[Abs[approx[x2q[#]] - func[#]]&, conslocs];
         
         Print[{"error ", CForm[error], " constraints ", CForm[consvals]}];
         Write[outfile, {func, ndegrees, denndegrees, CForm[error],
                         CForm[HornerForm[approx[x2q[x]]]]}]];

findcoeffs2[func_, xmin_, xmax_,
            dmin_, dstep_, maxndegrees_,
            dendmin_, dendstep_, maxdenndegrees_,
            conslocs_] :=
  Do[findcoeffs[func, xmin, xmax, dmin, dstep, ndegrees, denndegrees, conslocs],
     {ndegrees, maxndegrees},
     {denndegrees, dendmin, dendstep maxdenndegrees, dendstep}];

outfile = OpenWrite["/tmp/tmp"];
(* findcoeffs[Tan[Pi #]&, 0, 1/2,   1, 2, 3,   4, {0, 1/4}]; *)
(* findcoeffs[Tan[Pi #]&, 0, 1/2,   1, 2, 3,   4, {0, 1/4}]; *)
(* findcoeffs2[Tan[Pi #]&, 0, 1/2,   1, 2, 5,   2, 2, 5, {0, 1/4}]; *)

(*
PADE DOES NOT REMOVE SINGULARITIES
PROBABLY NEED TO DO THAT MANUALLY.
*)

(* findcoeffs[ArcTan[#]&, 0, 1,   1, 2, 10,   1, {0, 1}]; *)
(* findcoeffs2[ArcTan,   0, 1,   1, 2, 10,   1, 2, 10,   {0, 1}]; *)
(* findcoeffs2[ArcTan[#]&,   0, 1/2,   1, 2, 15,   1, 2, 1,   {0, 1/2}]; *)

(* findcoeffs2[Log[2, (1+#)/(1-#)]&,
            -3 + 2 Sqrt[2], 3 - 2 Sqrt[2],   0, 1, 15,   0, 1, 0,   {0}]; *)



outfile = OpenWrite["coeffs.out"];
Write[outfile, "(* Coefficients for function approximations *)"];
Map[findcoeffs2@@# &, funcs];
Close[outfile];
