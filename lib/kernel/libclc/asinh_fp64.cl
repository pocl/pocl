/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
 *
 * Copyright (c) 2017 Michal Babej / Tampere University of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */



#define NA0  (vtype)-0.12845379283524906084997e0
#define NA1  (vtype)-0.21060688498409799700819e0
#define NA2  (vtype)-0.10188951822578188309186e0
#define NA3  (vtype)-0.13891765817243625541799e-1
#define NA4  (vtype)-0.10324604871728082428024e-3

#define DA0  (vtype)0.77072275701149440164511e0
#define DA1  (vtype)0.16104665505597338100747e1
#define DA2  (vtype)0.11296034614816689554875e1
#define DA3  (vtype)0.30079351943799465092429e0
#define DA4  (vtype)0.235224464765951442265117e-1

#define NB0  (vtype)-0.12186605129448852495563e0
#define NB1  (vtype)-0.19777978436593069928318e0
#define NB2  (vtype)-0.94379072395062374824320e-1
#define NB3  (vtype)-0.12620141363821680162036e-1
#define NB4  (vtype)-0.903396794842691998748349e-4

#define DB0  (vtype)0.73119630776696495279434e0
#define DB1  (vtype)0.15157170446881616648338e1
#define DB2  (vtype)0.10524909506981282725413e1
#define DB3  (vtype)0.27663713103600182193817e0
#define DB4  (vtype)0.21263492900663656707646e-1

#define NC0  (vtype)-0.81210026327726247622500e-1
#define NC1  (vtype)-0.12327355080668808750232e0
#define NC2  (vtype)-0.53704925162784720405664e-1
#define NC3  (vtype)-0.63106739048128554465450e-2
#define NC4  (vtype)-0.35326896180771371053534e-4

#define DC0  (vtype)0.48726015805581794231182e0
#define DC1  (vtype)0.95890837357081041150936e0
#define DC2  (vtype)0.62322223426940387752480e0
#define DC3  (vtype)0.15028684818508081155141e0
#define DC4  (vtype)0.10302171620320141529445e-1

#define ND0  (vtype)-0.4638179204422665073e-1
#define ND1  (vtype)-0.7162729496035415183e-1
#define ND2  (vtype)-0.3247795155696775148e-1
#define ND3  (vtype)-0.4225785421291932164e-2
#define ND4  (vtype)-0.3808984717603160127e-4
#define ND5  (vtype)0.8023464184964125826e-6

#define DD0  (vtype)0.2782907534642231184e0
#define DD1  (vtype)0.5549945896829343308e0
#define DD2  (vtype)0.3700732511330698879e0
#define DD3  (vtype)0.9395783438240780722e-1
#define DD4  (vtype)0.7200057974217143034e-2

#define NE0  (vtype)-0.121224194072430701e-4
#define NE1  (vtype)-0.273145455834305218e-3
#define NE2  (vtype)-0.152866982560895737e-2
#define NE3  (vtype)-0.292231744584913045e-2
#define NE4  (vtype)-0.174670900236060220e-2
#define NE5  (vtype)-0.891754209521081538e-12

#define DE0  (vtype)0.499426632161317606e-4
#define DE1  (vtype)0.139591210395547054e-2
#define DE2  (vtype)0.107665231109108629e-1
#define DE3  (vtype)0.325809818749873406e-1
#define DE4  (vtype)0.415222526655158363e-1
#define DE5  (vtype)0.186315628774716763e-1

#define NF0   (vtype)-0.195436610112717345e-4
#define NF1   (vtype)-0.233315515113382977e-3
#define NF2   (vtype)-0.645380957611087587e-3
#define NF3   (vtype)-0.478948863920281252e-3
#define NF4   (vtype)-0.805234112224091742e-12
#define NF5   (vtype)0.246428598194879283e-13

#define DF0   (vtype)0.822166621698664729e-4
#define DF1   (vtype)0.135346265620413852e-2
#define DF2   (vtype)0.602739242861830658e-2
#define DF3   (vtype)0.972227795510722956e-2
#define DF4   (vtype)0.510878800983771167e-2

#define NG0   (vtype)-0.209689451648100728e-6
#define NG1   (vtype)-0.219252358028695992e-5
#define NG2   (vtype)-0.551641756327550939e-5
#define NG3   (vtype)-0.382300259826830258e-5
#define NG4   (vtype)-0.421182121910667329e-17
#define NG5   (vtype)0.492236019998237684e-19

#define DG0   (vtype)0.889178444424237735e-6
#define DG1   (vtype)0.131152171690011152e-4
#define DG2   (vtype)0.537955850185616847e-4
#define DG3   (vtype)0.814966175170941864e-4
#define DG4   (vtype)0.407786943832260752e-4

#define NH0   (vtype)-0.178284193496441400e-6
#define NH1   (vtype)-0.928734186616614974e-6
#define NH2   (vtype)-0.923318925566302615e-6
#define NH3   (vtype)-0.776417026702577552e-19
#define NH4   (vtype)0.290845644810826014e-21

#define DH0   (vtype)0.786694697277890964e-6
#define DH1   (vtype)0.685435665630965488e-5
#define DH2   (vtype)0.153780175436788329e-4
#define DH3   (vtype)0.984873520613417917e-5

#define NI0   (vtype)-0.538003743384069117e-10
#define NI1   (vtype)-0.273698654196756169e-9
#define NI2   (vtype)-0.268129826956403568e-9
#define NI3   (vtype)-0.804163374628432850e-29

#define DI0   (vtype)0.238083376363471960e-9
#define DI1   (vtype)0.203579344621125934e-8
#define DI2   (vtype)0.450836980450693209e-8
#define DI3   (vtype)0.286005148753497156e-8

_CL_OVERLOADABLE vtype asinh(vtype x) {
    const vtype rteps = (vtype)0x1.6a09e667f3bcdp-27;
    const vtype recrteps = (vtype)0x1.6a09e667f3bcdp+26;

    // log2_lead and log2_tail sum to an extra-precise version of log(2)
    const vtype log2_lead = (vtype)0x1.62e42ep-1;
    const vtype log2_tail = (vtype)0x1.efa39ef35793cp-25;

    utype ux = as_utype(x);
    utype ax = ux & (utype)~SIGNBIT_DP64;
    vtype absx = as_vtype(ax);

    vtype t = x * x;
    vtype pn, tn, pd, td;

    // XXX we are betting here that we can evaluate 8 pairs of
    // polys faster than we can grab 12 coefficients from a table
    // This also uses fewer registers

    // |x| >= 8
    pn = pocl_fma(t, pocl_fma(t, pocl_fma(t, NI3, NI2), NI1), NI0);
    pd = pocl_fma(t, pocl_fma(t, pocl_fma(t, DI3, DI2), DI1), DI0);

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NH4, NH3), NH2), NH1), NH0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, DH3, DH2), DH1), DH0);
    pn = (absx < (vtype)8.0) ? tn : pn;
    pd = (absx < (vtype)8.0) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NG5, NG4), NG3), NG2), NG1), NG0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DG4, DG3), DG2), DG1), DG0);
    pn = (absx < (vtype)4.0) ? tn : pn;
    pd = (absx < (vtype)4.0) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NF5, NF4), NF3), NF2), NF1), NF0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DF4, DF3), DF2), DF1), DF0);
    pn = (absx < (vtype)2.0) ? tn : pn;
    pd = (absx < (vtype)2.0) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NE5, NE4), NE3), NE2), NE1), NE0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DE5, DE4), DE3), DE2), DE1), DE0);
    pn = (absx < (vtype)1.5) ? tn : pn;
    pd = (absx < (vtype)1.5) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, ND5, ND4), ND3), ND2), ND1), ND0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DD4, DD3), DD2), DD1), DD0);
    pn = (absx <= (vtype)1.0) ? tn : pn;
    pd = (absx <= (vtype)1.0) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NC4, NC3), NC2), NC1), NC0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DC4, DC3), DC2), DC1), DC0);
    pn = (absx < (vtype)0.75) ? tn : pn;
    pd = (absx < (vtype)0.75) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NB4, NB3), NB2), NB1), NB0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DB4, DB3), DB2), DB1), DB0);
    pn = (absx < (vtype)0.5) ? tn : pn;
    pd = (absx < (vtype)0.5) ? td : pd;

    tn = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, NA4, NA3), NA2), NA1), NA0);
    td = pocl_fma(t, pocl_fma(t, pocl_fma(t, pocl_fma(t, DA4, DA3), DA2), DA1), DA0);
    pn = (absx < (vtype)0.25) ? tn : pn;
    pd = (absx < (vtype)0.25) ? td : pd;

    vtype pq = MATH_DIVIDE(pn, pd);

    // |x| <= 1
    vtype result1 = pocl_fma(absx*t, pq, absx);

    // Other ranges
    itype xout = (absx <= (vtype)32.0) | (absx > recrteps);
    vtype y = absx + sqrt(pocl_fma(absx, absx, (vtype)1.0));
    y = xout ? absx : y;

    vtype r1, r2;
    itype xexp;
    __pocl_ep_log(y, &xexp, &r1, &r2);

    itype xout2 = (xout ? (itype)1 : (itype)0);
    vtype dxexp = convert_vtype(xexp + xout2);
    r1 = pocl_fma(dxexp, log2_lead, r1);
    r2 = pocl_fma(dxexp, log2_tail, r2);

    // 1 < x <= 32
    vtype v2 = (pq + (vtype)0.25) / t;
    vtype r = v2 + r1;
    vtype s = ((r1 - r) + v2) + r2;
    vtype v1 = r + s;
    v2 = (r - v1) + s;
    vtype result2 = v1 + v2;

    // x > 32
    vtype result3 = r1 + r2;

    vtype ret = (absx > (vtype)1.0) ? result2 : result1;
    ret = (absx > (vtype)32.0) ? result3 : ret;
    ret = (x < (vtype)0.0) ? -ret : ret;

    // NaN, +-Inf, or x small enough that asinh(x) = x
    ret = ((ax >= (utype)PINFBITPATT_DP64) | (absx < rteps)) ? x : ret;
    return ret;
}
