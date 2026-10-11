"""楕円曲線・保型形式・L関数を学ぶための計算部品。

一般 Weierstrass 方程式上の群演算は Fraction、形式的 q 級数は
Python の任意精度整数 / Fraction で計算する。近似計算は明示的に分離する。
Tate アルゴリズム、一般の Selmer 群計算、一般のランク証明は実装していない。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import gcd, isqrt, log, ceil, factorial
from numbers import Integral
from typing import Sequence, Callable
import math
import shutil
import subprocess
import time
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.special import roots_legendre, rgamma

Point = tuple[F, F] | None


def rational(value) -> F:
    """二進浮動小数点を拒否し、有理数への意図しない丸めを防ぐ。"""
    if isinstance(value, (float, np.floating)):
        raise TypeError("厳密計算には float ではなく Fraction / 整数を指定してください。")
    if isinstance(value, sp.Rational):
        return F(int(value.p), int(value.q))
    if isinstance(value, Integral):
        return F(int(value))
    return F(value)


def qpoint(x, y) -> Point:
    return rational(x), rational(y)


def point_text(P: Point) -> str:
    return "O" if P is None else f"({P[0]}, {P[1]})"


@dataclass(frozen=True)
class EllipticCurve:
    """y²+a1*x*y+a3*y=x³+a2*x²+a4*x+a6。無限遠点は None。"""
    a1: F = F(0)
    a2: F = F(0)
    a3: F = F(0)
    a4: F = F(0)
    a6: F = F(0)

    def __post_init__(self):
        for name in ("a1", "a2", "a3", "a4", "a6"):
            object.__setattr__(self, name, rational(getattr(self, name)))
        if self.invariants()["discriminant"] == 0:
            raise ValueError("判別式が 0 です。特異三次曲線は楕円曲線ではありません。")

    @property
    def ainvs(self) -> tuple[F, ...]:
        return self.a1, self.a2, self.a3, self.a4, self.a6

    def invariants(self) -> dict[str, F]:
        a1, a2, a3, a4, a6 = self.ainvs
        b2 = a1*a1 + 4*a2
        b4 = a1*a3 + 2*a4
        b6 = a3*a3 + 4*a6
        b8 = a1*a1*a6 + 4*a2*a6 - a1*a3*a4 + a2*a3*a3 - a4*a4
        c4 = b2*b2 - 24*b4
        c6 = -b2**3 + 36*b2*b4 - 216*b6
        disc = -b2*b2*b8 - 8*b4**3 - 27*b6*b6 + 9*b2*b4*b6
        return dict(b2=b2, b4=b4, b6=b6, b8=b8, c4=c4, c6=c6,
                    discriminant=disc, j=c4**3/disc if disc else None)

    def contains(self, P: Point) -> bool:
        if P is None:
            return True
        x, y = map(rational, P)
        return y*y+self.a1*x*y+self.a3*y == x**3+self.a2*x*x+self.a4*x+self.a6

    def point(self, x, y) -> Point:
        P = qpoint(x, y)
        if not self.contains(P):
            raise ValueError(f"{P} はこの曲線上にありません。")
        return P

    def neg(self, P: Point) -> Point:
        if not self.contains(P):
            raise ValueError("曲線上にない点です。")
        if P is None:
            return None
        x, y = map(rational, P)
        return x, -y-self.a1*x-self.a3

    def add(self, P: Point, Q: Point) -> Point:
        if not self.contains(P) or not self.contains(Q):
            raise ValueError("曲線上にない点です。")
        if P is None:
            return Q
        if Q is None:
            return P
        x1, y1 = map(rational, P)
        x2, y2 = map(rational, Q)
        if x1 == x2:
            if y1+y2+self.a1*x1+self.a3 == 0:
                return None
            den = 2*y1+self.a1*x1+self.a3
            slope = (3*x1*x1+2*self.a2*x1+self.a4-self.a1*y1)/den
        else:
            slope = (y2-y1)/(x2-x1)
        intercept = y1-slope*x1
        x3 = slope*slope+self.a1*slope-self.a2-x1-x2
        y3 = -(slope+self.a1)*x3-intercept-self.a3
        return x3, y3

    def mul(self, n: int, P: Point) -> Point:
        if not isinstance(n, Integral):
            raise TypeError("倍率には整数が必要です。")
        n = int(n)
        if not self.contains(P):
            raise ValueError("曲線上にない点です。")
        if n < 0:
            return self.mul(-n, self.neg(P))
        R = None
        while n:
            if n & 1:
                R = self.add(R, P)
            n >>= 1
            if n:
                P = self.add(P, P)
        return R

    def order_up_to(self, P: Point, bound: int = 12) -> int | None:
        """None は上限内で O に到達しなかったことだけを表す。"""
        if bound < 1:
            raise ValueError("bound は正です。")
        if P is None:
            return 1
        Q = None
        for n in range(1, bound+1):
            Q = self.add(Q, P)
            if Q is None:
                return n
        return None

    def short_model(self) -> tuple[EllipticCurve, Callable[[Point], Point]]:
        """Q 上で平方完成・平行移動する。最小モデル化ではない。"""
        iv = self.invariants()
        short = EllipticCurve(a4=-iv["c4"]/48, a6=-iv["c6"]/864)
        def transform(P):
            if P is None:
                return None
            x, y = P
            return x+iv["b2"]/12, y+(self.a1*x+self.a3)/2
        return short, transform

    def residue_ainvs(self, p: int) -> tuple[int, ...]:
        if not sp.isprime(p):
            raise ValueError("p は素数でなければなりません。")
        values = []
        for a in self.ainvs:
            if a.denominator % p == 0:
                raise ValueError("係数の分母が p で割れます。整モデルが必要です。")
            values.append(a.numerator*pow(a.denominator, -1, p) % p)
        return tuple(values)

    def smooth_mod_p(self, p: int) -> bool:
        self.residue_ainvs(p)
        d = self.invariants()["discriminant"]
        return d.numerator % p != 0

    def points_mod_p(self, p: int) -> list[tuple[int, int] | None]:
        """O(p²) の参照用総当たり。特異な還元もそのまま数える。"""
        a1, a2, a3, a4, a6 = self.residue_ainvs(p)
        return [None] + [(x, y) for x in range(p) for y in range(p)
                         if (y*y+a1*x*y+a3*y-x**3-a2*x*x-a4*x-a6) % p == 0]

    @lru_cache(maxsize=8192)
    def count_mod_p(self, p: int) -> int:
        """平方剰余表を用いる O(p) の計数。p=2 は総当たり。"""
        a1, a2, a3, a4, a6 = self.residue_ainvs(p)
        if p == 2:
            return len(self.points_mod_p(p))
        # float / int64 のオーバーフローに依存しない実装。
        chi = [-1]*p
        chi[0] = 0
        for y in range(1, (p+1)//2):
            chi[y*y % p] = 1
        trace = sum(chi[((a1*x+a3)**2 + 4*(x**3+a2*x*x+a4*x+a6)) % p]
                    for x in range(p))
        return p+1+trace

    def ap_good(self, p: int) -> int:
        if not self.smooth_mod_p(p):
            raise ValueError("このモデルの還元は特異です。良い素数の式を使えません。")
        return p+1-self.count_mod_p(p)

    def singular_points_mod_p(self, p: int) -> list[tuple[int, int]]:
        a1, a2, a3, a4, _ = self.residue_ainvs(p)
        return [(x, y) for P in self.points_mod_p(p) if P is not None
                for x, y in [P]
                if (a1*y-3*x*x-2*a2*x-a4) % p == 0
                and (2*y+a1*x+a3) % p == 0]

    def tangent_slopes_mod_p(self, p: int, P: tuple[int, int]) -> list[int]:
        a1, a2, *_ = self.residue_ainvs(p)
        x, _ = P
        return [t for t in range(p) if (t*t+a1*t-3*x-a2) % p == 0]


def rational_sqrt(v: F) -> F | None:
    v = rational(v)
    if v < 0:
        return None
    a, b = isqrt(v.numerator), isqrt(v.denominator)
    return F(a, b) if a*a == v.numerator and b*b == v.denominator else None


def integral_points_short(E: EllipticCurve, xmin: int, xmax: int) -> list[Point]:
    if E.a1 or E.a3:
        raise ValueError("この探索関数は y²=三次式 専用です。")
    out = []
    for x in range(xmin, xmax+1):
        y = rational_sqrt(x**3+E.a2*x*x+E.a4*x+E.a6)
        if y is not None and y.denominator == 1:
            out.append(E.point(x, y))
            if y:
                out.append(E.point(x, -y))
    return out


def torsion_generated(E: EllipticCurve, generators: Sequence[Point], cap: int = 32) -> set[Point]:
    points = {None}
    todo = [None]
    while todo:
        P = todo.pop()
        for Q in generators:
            R = E.add(P, Q)
            if R not in points:
                points.add(R)
                todo.append(R)
                if len(points) > cap:
                    raise ValueError("生成部分群が cap を超えました。有限群とは確認できません。")
    return points


def congruent_triangle(n: int, P: Point, positive: bool = True) -> tuple[F, F, F]:
    E = EllipticCurve(a4=-n*n)
    if n <= 0 or P is None or not E.contains(P) or P[1] == 0:
        raise ValueError("n>0 と En 上の y≠0 の点が必要です。")
    x, y = P
    a, b, c = (x*x-n*n)/y, 2*n*x/y, (x*x+n*n)/y
    return tuple(map(abs, (a, b, c))) if positive else (a, b, c)


def triangle_to_point(n: int, a, b, c) -> Point:
    a, b, c = map(rational, (a, b, c))
    if a*a+b*b != c*c or a*b/2 != n or c == a:
        raise ValueError("符号つき有理直角三角形の条件を満たしません。")
    return EllipticCurve(a4=-n*n).point(n*b/(c-a), 2*n*n/(c-a))


def weighted_ternary_count(n: int, weights: tuple[int, int, int]) -> int:
    if n < 0 or any(w <= 0 for w in weights):
        raise ValueError("n≥0 と正の重みが必要です。")
    a, b, c = weights
    total = 0
    for x in range(-isqrt(n//a), isqrt(n//a)+1):
        rem = n-a*x*x
        for y in range(-isqrt(rem//b), isqrt(rem//b)+1):
            z2, r = divmod(rem-b*y*y, c)
            if r == 0 and isqrt(z2)**2 == z2:
                total += 1 if z2 == 0 else 2
    return total


def tunnell_test(n: int) -> dict:
    if n <= 0 or any(e > 1 for e in sp.factorint(n).values()):
        raise ValueError("正の平方因子を持たない n が必要です。")
    target, a = (n, 2) if n % 2 else (n//2, 4)
    left = weighted_ternary_count(target, (a, 1, 32))
    right_count = weighted_ternary_count(target, (a, 1, 8))
    return dict(n=n, A=left, B=right_count, passes=(2*left == right_count))


def divisor_sums(N: int, power: int = 1) -> list[int]:
    if N < 0 or power < 0:
        raise ValueError("N と power は非負です。")
    out = [0]*(N+1)
    for d in range(1, N+1):
        w = d**power
        for j in range(d, N+1, d):
            out[j] += w
    return out


def series_mul(a: Sequence, b: Sequence, N: int) -> list:
    if N < 0:
        raise ValueError("N は非負です。")
    out = [0]*(N+1)
    for i, x in enumerate(a[:N+1]):
        if x:
            for j, y in enumerate(b[:N+1-i]):
                if y:
                    out[i+j] += x*y
    return out


def series_power(a: Sequence, k: int, N: int) -> list:
    if k < 0:
        raise ValueError("非負整数乗のみ対応します。")
    out = [1]+[0]*N
    a = list(a)
    while k:
        if k & 1:
            out = series_mul(out, a, N)
        k >>= 1
        if k:
            a = series_mul(a, a, N)
    return out


def series_div(a: Sequence, b: Sequence, N: int) -> list[F]:
    if len(a) <= N or len(b) <= N or not b[0]:
        raise ValueError("定数項が非零の除数と、必要な次数までの係数が必要です。")
    out = [F(0)]*(N+1)
    for n in range(N+1):
        out[n] = (rational(a[n])-sum(rational(b[j])*out[n-j] for j in range(1,n+1)))/rational(b[0])
    return out


def theta_power(k: int, N: int) -> list[int]:
    theta = [0]*(N+1)
    theta[0] = 1
    for j in range(1, isqrt(N)+1):
        theta[j*j] = 2
    return series_power(theta, k, N)


def eisenstein_coefficients(weight: int, N: int) -> list[F]:
    if weight < 4 or weight % 2:
        raise ValueError("SL2(Z) の正規化 Eisenstein 級数はここでは偶数 weight≥4 に限定。")
    factor = -F(2*weight)/rational(sp.bernoulli(weight))
    return [F(1)]+[factor*x for x in divisor_sums(N, weight-1)[1:]]


def eta_product_coefficients(exponents: dict[int, int], N: int) -> list[int]:
    """積の対数微分から厳密係数を計算。整数かつ非負の先頭次数に限定。"""
    if not exponents or N < 0 or any(d <= 0 or not isinstance(r, Integral) for d,r in exponents.items()):
        raise ValueError("正の整数 d、整数指数 r、非負次数が必要です。")
    shift24 = sum(d*r for d,r in exponents.items())
    if shift24 < 0 or shift24 % 24:
        raise ValueError("この実装は整数かつ非負の q 先頭次数に限定します。")
    shift = shift24//24
    if shift > N:
        return [0]*(N+1)
    M = N-shift
    sigma = divisor_sums(M)
    b = [0]*(M+1)
    for d,r in exponents.items():
        for n in range(d, M+1, d):
            b[n] -= r*d*sigma[n//d]
    c = [1]+[0]*M
    for n in range(1,M+1):
        numerator = sum(b[j]*c[n-j] for j in range(1,n+1))
        c[n], rem = divmod(numerator,n)
        if rem:
            raise ArithmeticError("eta 積係数の整数性に反しました。")
    return [0]*shift+c


def eta_product_direct(exponents: dict[int, int], N: int) -> list[int]:
    """正指数のみの有限積。対数微分実装の独立な検査用。"""
    if any(r < 0 for r in exponents.values()):
        raise ValueError("直接積実装は非負指数のみです。")
    shift24 = sum(d*r for d,r in exponents.items())
    if shift24 < 0 or shift24 % 24:
        raise ValueError("先頭次数は非負整数に限定。")
    shift = shift24//24
    if shift > N:
        return [0]*(N+1)
    M = N-shift
    a = [1]+[0]*M
    for d,r in exponents.items():
        for step in range(d,M+1,d):
            for _ in range(r):
                for j in range(M, step-1,-1):
                    a[j] -= a[j-step]
    return [0]*shift+a


def qseries_text(coefficients: Sequence, count: int = 16) -> str:
    q = sp.Symbol("q")
    count = min(count,len(coefficients)-1)
    expr = sum(sp.Rational(c.numerator,c.denominator)*q**i if isinstance(c,F) else c*q**i
               for i,c in enumerate(coefficients[:count+1]))
    return str(expr) + f" + O(q^{count+1})"


def q_eval(coefficients: Sequence, tau, dps: int = 50):
    with mp.workdps(dps):
        tau = mp.mpc(tau)
        if tau.imag <= 0:
            raise ValueError("tau は上半平面にある必要があります。")
        q = mp.exp(2*mp.pi*1j*tau)
        acc = mp.mpc(0)
        for c in reversed(coefficients):
            cc = mp.mpf(c.numerator)/c.denominator if isinstance(c,F) else mp.mpf(c)
            acc = acc*q+cc
        return +acc


def dimension_level1(weight: int) -> int:
    if weight < 0 or weight % 2:
        return 0
    return weight//12+(0 if weight % 12 == 2 else 1)


def gamma0_invariants(N: int) -> dict:
    if N <= 0:
        raise ValueError("N は正です。")
    index = F(N)
    for p in sp.factorint(N):
        index *= F(int(p)+1,int(p))
    cusps = sum(int(sp.totient(gcd(int(d),N//int(d)))) for d in sp.divisors(N))
    e2 = sum((x*x+1)%N == 0 for x in range(N))
    e3 = sum((x*x+x+1)%N == 0 for x in range(N))
    genus = 1+index/12-F(e2,4)-F(e3,3)-F(cusps,2)
    if genus.denominator != 1:
        raise ArithmeticError("種数が整数になりません。")
    return dict(N=N,index=int(index),cusps=cusps,e2=e2,e3=e3,genus=int(genus))


def sturm_bound(N: int, weight: int) -> int:
    """両辺が同じ保型形式空間に属することが既知の場合の係数比較の閾値。"""
    return weight*gamma0_invariants(N)["index"]//12


def hecke_prime(a: Sequence, p: int, weight: int, level: int, out_degree: int) -> list:
    if not sp.isprime(p) or out_degree < 0 or p*out_degree >= len(a):
        raise ValueError("素数 p と、p*out_degree までの既知係数が必要です。")
    character = 0 if level % p == 0 else 1
    return [a[p*n]+(character*p**(weight-1)*a[n//p] if n % p == 0 else 0)
            for n in range(out_degree+1)]


@dataclass(frozen=True)
class CurveData:
    """導手・符号・悪い素数の局所係数は明示入力。自動推定しない。"""
    name: str
    curve: EllipticCurve
    conductor: int
    root_number: int
    bad_ap: tuple[tuple[int,int], ...]
    source: str

    def __post_init__(self):
        if self.conductor <= 0 or self.root_number not in (-1,1):
            raise ValueError("導手と root number の入力が不正です。")
        d = self.curve.invariants()["discriminant"]
        if any(a.denominator != 1 for a in self.curve.ainvs):
            raise ValueError("本実装の L 級数データには整係数モデルが必要です。")
        if set(sp.factorint(abs(d.numerator))) != {p for p,a in self.bad_ap}:
            raise ValueError("この整モデルの全ての悪い素数について局所係数を指定してください。")
        if any(not sp.isprime(p) or a not in (-1,0,1) for p,a in self.bad_ap):
            raise ValueError("bad_ap が不正です。")

    def ap(self, p: int) -> int:
        bad = dict(self.bad_ap)
        return bad[p] if p in bad else self.curve.ap_good(p)


E11 = EllipticCurve(0,-1,1,-10,-20)
E37A = EllipticCurve(0,0,1,-1,0)
E37B = EllipticCurve(0,1,1,-23,-50)
E389 = EllipticCurve(0,1,1,-2,0)
E5077 = EllipticCurve(0,0,1,-7,6)
CURVES = {
    "11a1": CurveData("11a1",E11,11,1,((11,1),),"https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/lseries_ell.html"),
    "37a1": CurveData("37a1",E37A,37,-1,((37,-1),),"https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/lseries_ell.html"),
    "37b1": CurveData("37b1",E37B,37,1,((37,1),),"https://doc.sagemath.org/html/en/reference/arithmetic_curves/sage/schemes/elliptic_curves/lseries_ell.html"),
    "389a1": CurveData("389a1",E389,389,1,((389,1),),"最小判別式 389、非分裂でない乗法的還元 a_389=1、w=1"),
    "5077a1": CurveData("5077a1",E5077,5077,-1,((5077,-1),),"https://www.lmfdb.org/EllipticCurve/Q/5077/a/1"),
}


@lru_cache(maxsize=32)
def l_coefficients(data: CurveData, N: int) -> tuple[int, ...]:
    """添字0は0。悪い素数と良い素数の再帰を分ける。"""
    if N < 1:
        raise ValueError("N≥1 が必要です。")
    a = [0]*(N+1)
    a[1] = 1
    bad = dict(data.bad_ap)
    spf = list(range(N+1))
    for p in range(2,isqrt(N)+1):
        if spf[p] == p:
            for j in range(p*p,N+1,p):
                if spf[j] == j:
                    spf[j] = p
    aps = {p:data.ap(p) for p in range(2,N+1) if spf[p] == p}
    for n in range(2,N+1):
        p = spf[n]
        m = n//p
        if m % p:
            a[n] = aps[p]*a[m]
        else:
            a[n] = aps[p]*a[m]-(0 if p in bad else p*a[m//p])
    return tuple(a)


def dirichlet_partial(a: Sequence[int], s):
    return sum(a[n]*n**(-s) for n in range(1,len(a)))


def euler_partial(data: CurveData, s, prime_bound: int):
    out = 1+0j if isinstance(s,complex) else 1.0
    bad = dict(data.bad_ap)
    for p in map(int,sp.primerange(2,prime_bound+1)):
        x = p**(-s)
        out /= 1-data.ap(p)*x+(0 if p in bad else p*x*x)
    return out


def central_L(data: CurveData, N: int = 200, dps: int = 60):
    """L(1) と L'(1) (w=-1 の場合)。tail_bound は丸め誤差を含まない。"""
    a = l_coefficients(data,N)
    with mp.workdps(dps):
        c = 2*mp.pi/mp.sqrt(data.conductor)
        r = mp.exp(-c)
        # |a_n|≤d(n)√n≤2n による保守的な数学的打切り上界。
        base_tail = 4*r**(N+1)/(1-r)
        if data.root_number == 1:
            value = 2*mp.fsum(mp.mpf(a[n])/n*mp.exp(-c*n) for n in range(1,N+1))
            return dict(kind="L(1)",value=+value,tail_bound=+base_tail,terms=N,dps=dps)
        value = 2*mp.fsum(mp.mpf(a[n])/n*mp.e1(c*n) for n in range(1,N+1))
        return dict(kind="L'(1)",value=+value,tail_bound=+(base_tail/(c*(N+1))),terms=N,dps=dps)


class MellinQuadrature:
    """分割 Mellin 積分を Gauss-Legendre 求積する。倍精度、厳密保証なし。

    関数等式の対称性を式に組み込むため、この式の対称性検査は
    関数等式の独立な検証にはならない。
    """
    def __init__(self, data: CurveData, terms: int = 700, nodes: int = 400, decay: float = 48):
        if terms < 1 or nodes < 16 or decay <= 0:
            raise ValueError("計算パラメータが不正です。")
        self.data = data
        self.c = 2*np.pi/np.sqrt(data.conductor)
        self.terms, self.nodes, self.decay = terms,nodes,decay
        self.T = 1+decay/self.c
        x,w = roots_legendre(nodes)
        self.t = 1+(x+1)*(self.T-1)/2
        self.w = w*(self.T-1)/2
        a = np.asarray(l_coefficients(data,terms)[1:],dtype=float)
        n = np.arange(1,terms+1,dtype=float)
        self.ft = np.exp(-self.c*np.outer(self.t,n))@a
        self.weighted = self.w*self.ft
        self.logt = np.log(self.t)

    def completed(self, s):
        return np.sum(self.weighted*(np.exp((s-1)*self.logt)
                      + self.data.root_number*np.exp((1-s)*self.logt)))

    def value(self, s):
        return self.c**s*rgamma(s)*self.completed(s)

    def central_coefficient(self, r: int):
        """Λ(1+z) の z^r 係数。L の係数とはまだ異なる。"""
        if r < 0:
            raise ValueError("r は非負です。")
        return ((1+self.data.root_number*(-1)**r)/factorial(r)
                *np.sum(self.weighted*self.logt**r))


def completed_gamma_sum(data: CurveData, s, N: int = 100, dps: int = 45):
    """不完全ガンマ関数による別実装。総和打切り以外は任意精度。"""
    a = l_coefficients(data,N)
    with mp.workdps(dps):
        s = mp.mpc(s)
        c = 2*mp.pi/mp.sqrt(data.conductor)
        return +mp.fsum(a[n]*((c*n)**(-s)*mp.gammainc(s,c*n,mp.inf)
                         +data.root_number*(c*n)**(s-2)*mp.gammainc(2-s,c*n,mp.inf))
                       for n in range(1,N+1) if a[n])


def real_period_one_component(E: EllipticCurve, dps: int = 55):
    """実根が1つの場合の ∫E(R)|dx/(2y+a1x+a3)|。最小整モデルが前提。"""
    iv = E.invariants()
    if iv["discriminant"] >= 0:
        raise ValueError("この関数は実連結成分が1つの場合だけに限定します。")
    with mp.workdps(dps):
        cv = lambda v: mp.mpf(v.numerator)/v.denominator
        b2,b4,b6 = [cv(iv[k]) for k in ("b2","b4","b6")]
        roots = mp.polyroots([4,b2,2*b4,b6],maxsteps=300,error=False)
        r = [z.real for z in roots if abs(mp.im(z)) < mp.mpf(10)**(-dps//2)][0]
        A = 3*r+b2/4
        B = 3*r*r+b2*r/2+b4/2
        return +(2*mp.quad(lambda t: 1/mp.sqrt(t**4+A*t*t+B),[0,1,mp.inf]))


def height_sequence(E: EllipticCurve, P: Point, depth: int = 6) -> list[dict]:
    """この実装の hhat=1/2 lim h(x(2^n P))/4^n。近似誤差保証はしない。"""
    if depth < 0 or not E.contains(P):
        raise ValueError("depth または点が不正です。")
    rows = []
    for n in range(depth+1):
        height = 0.0 if P is None else log(max(abs(P[0].numerator),P[0].denominator))
        bits = 0 if P is None else max(abs(P[0].numerator).bit_length(),P[0].denominator.bit_length())
        rows.append(dict(n=n,half_height=height/(2*4**n),full_height=height/4**n,bits=bits))
        if n < depth:
            P = E.add(P,P)
    return rows


def height_pairing_matrix(E: EllipticCurve, points: Sequence[Point], depth: int = 5) -> np.ndarray:
    """この実装の <P,Q>=hhat(P+Q)-hhat(P)-hhat(Q) の近似行列。"""
    @lru_cache(None)
    def height(P):
        return height_sequence(E,P,depth)[-1]["half_height"]
    return np.array([[height(E.add(P,Q))-height(P)-height(Q) for Q in points] for P in points])


def square_class(v) -> int:
    v = rational(v)
    if not v:
        raise ValueError("Q*/Q*² に 0 は入りません。")
    n = abs(v.numerator)*v.denominator
    out = -1 if v < 0 else 1
    for p,e in sp.factorint(n).items():
        if e % 2:
            out *= int(p)
    return out


def descent_delta(roots: tuple[int,int,int], P: Point) -> tuple[int,int,int]:
    if len(set(roots)) != 3:
        raise ValueError("相異なる3根が必要です。")
    if P is None:
        return (1,1,1)
    x,y = P
    if y*y != math.prod(x-e for e in roots):
        raise ValueError("指定された根を持つ曲線上にない点です。")
    out = []
    for i,e in enumerate(roots):
        v = math.prod(e-roots[j] for j in range(3) if j != i) if x == e else x-e
        out.append(square_class(v))
    return tuple(out)


def square_class_product(a: tuple[int,...],b: tuple[int,...]) -> tuple[int,...]:
    if len(a) != len(b):
        raise ValueError("次元が異なります。")
    return tuple(square_class(x*y) for x,y in zip(a,b))


def square_class_span(generators: Sequence[tuple[int,...]]) -> set[tuple[int,...]]:
    if not generators:
        return {(1,1,1)}
    span = {(1,)*len(generators[0])}
    for g in generators:
        span |= {square_class_product(x,g) for x in list(span)}
    return span


def half_point_by_interpolation(E: EllipticCurve, roots: tuple[int,int,int], Q: Point) -> Point:
    """平方類写像の核の逆構成 の構成。非2-torsion の Q と平方になる x(Q)-ei に限定。"""
    if E.a1 or E.a2 or E.a3 or sum(roots) or Q is None or Q[1] == 0:
        raise ValueError("短い Weierstrass 形・根の和0・非2-torsion点が必要です。")
    if not E.contains(Q):
        raise ValueError("曲線上にない点です。")
    sq = [rational_sqrt(Q[0]-e) for e in roots]
    if any(t is None for t in sq):
        raise ValueError("x(Q)-ei が全て有理数の平方になる必要があります。")
    x = sp.Symbol("x")
    for signs in product((-1,1),repeat=3):
        poly = sp.Poly(sp.interpolate([(e,sp.Rational(t.numerator,t.denominator)*s)
                        for e,t,s in zip(roots,sq,signs)],x),x)
        a,b,c = (rational(poly.nth(i)) for i in range(3))
        if c:
            P = E.point(b/c,1/c)
            if E.mul(2,P) == Q:
                return P
            if E.mul(2,E.neg(P)) == Q:
                return E.neg(P)
    raise ArithmeticError("構成した候補の倍点が一致しません。")


def primitive_obstruction_count(n: int, modulus: int = 8) -> int:
    """2Y²-X²=nW², Y²-Z²=nW² の primitive mod 2 解を数える。"""
    if modulus <= 0 or modulus & (modulus-1):
        raise ValueError("modulus は2の冪に限定します。")
    return sum((2*y*y-x*x-n*w*w)%modulus == 0 and (y*y-z*z-n*w*w)%modulus == 0
               for x,y,z,w in product(range(modulus),repeat=4) if (x|y|z|w)&1)


def hensel_simple(f: Callable[[int],int], df: Callable[[int],int], p: int,
                  root: int, depth: int = 6) -> list[int]:
    if not sp.isprime(p) or depth < 1 or f(root)%p or df(root)%p == 0:
        raise ValueError("単純根版 Hensel 補題の仮定を満たしません。")
    a, mod = root%p,p
    out = [a]
    for k in range(1,depth):
        correction = -(f(a)//mod)*pow(df(a),-1,p)%p
        a += correction*mod
        mod *= p
        a %= mod
        if f(a)%mod:
            raise ArithmeticError("Hensel lift に失敗。")
        out.append(a)
    return out


def valuation(n: int, p: int):
    if not sp.isprime(p):
        raise ValueError("p は素数です。")
    if n == 0:
        return math.inf
    n = abs(n)
    e = 0
    while n%p == 0:
        n//=p
        e+=1
    return e


def mobius(M: Sequence[int],tau: complex) -> complex:
    a,b,c,d = M
    if a*d-b*c != 1 or tau.imag <= 0:
        raise ValueError("SL2(Z) の行列と上半平面の点が必要です。")
    return (a*tau+b)/(c*tau+d)


def reduce_to_fundamental(tau: complex, max_steps: int = 100) -> tuple[complex,tuple[int,...],list[complex]]:
    if tau.imag <= 0 or not np.isfinite([tau.real,tau.imag]).all():
        raise ValueError("有限な上半平面の点が必要です。")
    z = complex(tau)
    M = np.eye(2,dtype=object)
    path = [z]
    for _ in range(max_steps):
        n = ceil(z.real-0.5)
        if n:
            z -= n
            M = np.array([[1,-n],[0,1]],dtype=object)@M
            path.append(z)
        if abs(z) < 1-1e-13 or (abs(abs(z)-1)<1e-13 and z.real < -1e-13):
            z = -1/z
            M = np.array([[0,-1],[1,0]],dtype=object)@M
            path.append(z)
        else:
            return z,tuple(int(v) for v in M.flat),path
    raise RuntimeError("基本領域への還元が max_steps 内に終了しません。")


def lattice_vectors(tau: complex, radius: int) -> np.ndarray:
    if tau.imag <= 0 or radius < 1:
        raise ValueError("上半平面の tau と radius≥1 が必要です。")
    return np.array([m*tau+n for m in range(-radius,radius+1)
                     for n in range(-radius,radius+1) if m or n],dtype=complex)


def lattice_eisenstein(weight: int,tau: complex,radius: int = 30):
    if weight < 4 or weight%2:
        raise ValueError("絶対収束する偶数重さ≥4 に限定。")
    return np.sum(lattice_vectors(tau,radius)**(-weight))


def weierstrass_wp(z: complex,tau: complex,radius: int = 30) -> tuple[complex,complex]:
    omega = lattice_vectors(tau,radius)
    if abs(z)<1e-14 or np.min(abs(z-omega))<1e-14:
        raise ValueError("格子点に極があります。")
    wp = 1/z**2+np.sum(1/(z-omega)**2-1/omega**2)
    derivative = -2/z**3-2*np.sum(1/(z-omega)**3)
    return wp,derivative


def petersson_delta(nodes: int = 48, ymax: float = 4.0, degree: int = 30) -> float:
    """SL2(Z) 上の正規化 Δ の Petersson norm²。倍精度の有限求積。"""
    if nodes < 8 or ymax <= 1 or degree < 1:
        raise ValueError("求積パラメータが不正です。")
    xs,xw = roots_legendre(nodes)
    ys,yw = roots_legendre(2*nodes)
    xs,xw = xs/2,xw/2
    coeff = np.array(eta_product_coefficients({1:24},degree),dtype=float)
    total = 0.0
    for x,wx in zip(xs,xw):
        lower = np.sqrt(1-x*x)
        y = lower+(ys+1)*(ymax-lower)/2
        q = np.exp(2j*np.pi*(x+1j*y))
        values = np.polynomial.polynomial.polyval(q,coeff)
        total += wx*np.sum(yw*(ymax-lower)/2*np.abs(values)**2*y**10)
    return float(total)


def dirichlet_character_partial(D: int, N: int = 10000) -> float:
    """負の基本判別式 D の二次 Dirichlet 指標の L(1) 部分和。

    Kronecker 記号を任意の負整数から作って周期配列にするのは誤り。
    基本判別式に限定すると conductor=|D| が正しく使える。
    N は項数。戻り値は倍精度の部分和で、尾の保証を含まない。
    """
    if not isinstance(D, Integral) or not isinstance(N, Integral) or D >= 0 or N < 1:
        raise ValueError("負の整数 D と正の項数 N が必要です。")
    D, N = int(D), int(N)
    if D % 4 == 1:
        fundamental = all(e == 1 for e in sp.factorint(abs(D)).values())
    elif D % 4 == 0:
        d = D // 4
        fundamental = d % 4 in (2, 3) and all(e == 1 for e in sp.factorint(abs(d)).values())
    else:
        fundamental = False
    if not fundamental:
        raise ValueError("D は負の基本判別式でなければなりません。例: -3, -4, -7, -8, -20。")
    period = abs(D)
    chi = np.array([int(sp.kronecker_symbol(D, n)) for n in range(period)])
    ns = np.arange(1, N + 1)
    return float(np.sum(chi[ns % period] / ns))


def pari_crosscheck(data: CurveData, terms: int = 30, timeout: int = 45) -> dict:
    """任意機能。gp が存在すれば固定した検証コマンドのみを実行する。"""
    gp = shutil.which("gp")
    if gp is None:
        return dict(status="SKIP",reason="PARI/GP の gp が未導入。主要解析には不要。")
    if terms<1 or terms>100000:
        raise ValueError("terms は 1..100000 に限定。")
    coeff = ",".join(str(a) for a in data.curve.ainvs)
    script = (f'default(realprecision,35); E=ellinit([{coeff}]); '
              f'print(ellglobalred(E)[1]); print(ellrootno(E)); '
              f'print(ellan(E,{int(terms)})); print(ellrank(E)[1..2]); quit;\n')
    run = subprocess.run([gp,"-fq"],input=script,text=True,capture_output=True,timeout=timeout)
    if run.returncode or "***" in run.stderr:
        raise RuntimeError(f"PARI/GP 検証に失敗: {run.stderr[:1000]}")
    lines = [x.strip() for x in run.stdout.splitlines() if x.strip()]
    import ast
    if len(lines)!=4:
        raise RuntimeError("予期しない PARI/GP 出力。")
    conductor,root = int(lines[0]),int(lines[1])
    ans, rank_bounds = ast.literal_eval(lines[2]),ast.literal_eval(lines[3])
    return dict(status="OK",conductor=conductor,root_number=root,
                coefficient_match=tuple(ans)==l_coefficients(data,terms)[1:],rank_bounds=rank_bounds)


def run_self_tests() -> dict:
    """独立な参照計算と境界条件を含む回帰テスト。assert 最適化の影響を受けない。"""
    total=0
    started=time.perf_counter()
    def check(condition,label):
        nonlocal total
        total+=1
        if not condition:
            raise AssertionError(label)
    def rejects(fn,label):
        try:
            fn()
        except (ValueError,TypeError):
            check(True,label)
        else:
            check(False,label)
    E=EllipticCurve(a6=1)
    P=E.point(2,3)
    expected=[None,P,E.point(0,1),E.point(-1,0),E.point(0,-1),E.point(2,-3),None]
    for n in range(7):
        check(E.mul(n,P)==expected[n],f"6-torsion {n}")
    check(E.mul(-1,P)==E.neg(P),"負の倍率")
    check(E.add(P,E.neg(P)) is None,"逆元")
    for i,j,k in product(range(6),repeat=3):
        A,B,C=expected[i],expected[j],expected[k]
        check(E.add(E.add(A,B),C)==E.add(A,E.add(B,C)),"結合則の有限検査")
    for curve in (E,E11,E37A,E37B,EllipticCurve(a4=-25)):
        iv=curve.invariants()
        check(iv["c4"]**3-iv["c6"]**2==1728*iv["discriminant"],"不変量恒等式")
        for p in (2,3,5,7,11,13,17,19):
            check(curve.count_mod_p(p)==len(curve.points_mod_p(p)),"点数の独立照合")
            if curve.smooth_mod_p(p):
                check(curve.ap_good(p)**2<=4*p,"Hasse")
    check(E11.invariants()["discriminant"]==-11**5,"11a1 判別式")
    Q=EllipticCurve(a4=-25).point(-4,6)
    tri=congruent_triangle(5,Q,False)
    check(triangle_to_point(5,*tri)==Q,"符号つき三角形逆写像")
    check(EllipticCurve(a4=-25).mul(2,Q)==qpoint(F(1681,144),F(-62279,1728)),"この実装の倍点")
    ec=l_coefficients(CURVES["11a1"],160)
    eta=eta_product_coefficients({1:2,11:2},160)
    check(list(ec)==eta,"EC と eta の独立係数一致")
    check(eta==eta_product_direct({1:2,11:2},160),"eta 積2実装")
    for p in (2,3,5,7,11):
        check(hecke_prime(eta,p,2,11,12)==[eta[p]*x for x in eta[:13]],"Hecke 特殊素数含む")
    N=80
    E4=eisenstein_coefficients(4,N); E6=eisenstein_coefficients(6,N)
    d=eta_product_coefficients({1:24},N)
    check(series_mul(E4,E6,N)==eisenstein_coefficients(10,N),"E4 E6 = E10")
    v4=series_power(E4,3,N);v6=series_power(E6,2,N)
    check([(a-b)/1728 for a,b in zip(v4,v6)]==d,"Delta 恒等式")
    check(all((d[n]-divisor_sums(N,11)[n])%691==0 for n in range(1,N+1)),"691 合同式")
    check(theta_power(4,N)[1:]==[8*sum(int(v) for v in sp.divisors(n) if v%4) for n in range(1,N+1)],"四平方")
    check(gamma0_invariants(37)["genus"]==2,"X0(37) 種数")
    check(sturm_bound(11,2)==2,"Sturm 11")
    check(primitive_obstruction_count(34)==0,"34 の局所障害")
    check(primitive_obstruction_count(82)==0,"82 の局所障害")
    roots=(6,20,-26);Ed=EllipticCurve(a4=-556,a6=3120)
    P,Q=Ed.point(-8,84),Ed.point(24,60)
    check(descent_delta(roots,Ed.add(P,Q))==square_class_product(descent_delta(roots,P),descent_delta(roots,Q)),"descent 準同型")
    H=half_point_by_interpolation(Ed,roots,Ed.mul(2,P))
    check(Ed.mul(2,H)==Ed.mul(2,P),"descent 逆構成")
    hs=hensel_simple(lambda x:x*x-2,lambda x:2*x,7,3,6)
    check(hs[:5]==[3,10,108,2166,4567],"定義に基づく Hensel")
    z,M,path=reduce_to_fundamental(2.7+.3j)
    check(abs(mobius(M,2.7+.3j)-z)<1e-12 and abs(z)>=1-1e-12,"基本領域")
    for n in (1,2,3):
        check(not tunnell_test(n)["passes"],"Tunnell 必要条件不成立")
    for n in (5,6,7):
        check(tunnell_test(n)["passes"],"Tunnell 必要条件成立")
    rejects(lambda:EllipticCurve(),"特異曲線拒否")
    rejects(lambda:E.point(1,1),"曲線外点拒否")
    rejects(lambda:E.count_mod_p(9),"合成数拒否")
    rejects(lambda:E11.ap_good(11),"悪い素数の誤用拒否")
    rejects(lambda:hecke_prime([0,1],2,2,11,10),"未知係数拒否")
    rejects(lambda:rational(.1),"float 拒否")
    rejects(lambda:tunnell_test(4),"平方因子あり拒否")
    rejects(lambda:square_class(0),"0 の平方類拒否")
    rejects(lambda:eisenstein_coefficients(2,10),"E2 誤用拒否")
    rejects(lambda:hensel_simple(lambda x:x*x-41,lambda x:2*x,2,1),"非単純根拒否")
    for D in (-3, -4, -7, -8, -11, -20):
        direct = sum(float(sp.kronecker_symbol(D, n))/n for n in range(1, 202))
        check(abs(dirichlet_character_partial(D, 201)-direct) < 1e-14,"Dirichlet 指標の周期性")
    rejects(lambda:dirichlet_character_partial(-5),"基本判別式でない入力の拒否")
    check(abs(dirichlet_character_partial(-4, 10000)-math.pi/4) < 1/10001,"beta 交代級数誤差")
    for row in verify_torsion_examples():
        check(row["total_order_candidates"] == str([row["generated_order"]]),"torsion の完全性")
    return dict(tests=total,status="PASS",seconds=round(time.perf_counter()-started,3))


def count_fp2_short(A: int, B: int, p: int = 5, nonresidue: int = 2) -> int:
    """Fp²=Fp[u]/(u²-d) を実装して短い方程式の点数を直接計数。"""
    if not sp.isprime(p) or p <= 3 or pow(nonresidue,(p-1)//2,p)!=p-1:
        raise ValueError("p>3 と平方非剰余 d が必要です。")
    def mul(x,y):
        return ((x[0]*y[0]+nonresidue*x[1]*y[1])%p,(x[0]*y[1]+x[1]*y[0])%p)
    elements=list(product(range(p),repeat=2))
    counts={x:0 for x in elements}
    for y in elements:
        counts[mul(y,y)]+=1
    total=1
    for x in elements:
        cube=mul(mul(x,x),x)
        total+=counts[((cube[0]+A*x[0]+B)%p,(cube[1]+A*x[1])%p)]
    return total


def modular_mellin(coefficients: Sequence, weight: int, level: int,
                    epsilon: int, s, nodes: int = 160, decay: float = 45):
    """一般重さの分割 Mellin 積分。保型性・Fricke固有性は入力の前提。"""
    if weight < 2 or level < 1 or epsilon not in (-1,1):
        raise ValueError("重さ・レベル・符号が不正です。")
    c=2*np.pi/np.sqrt(level)
    x,w=roots_legendre(nodes)
    T=1+decay/c
    t=1+(x+1)*(T-1)/2
    n=np.arange(1,len(coefficients),dtype=float)
    a=np.array([float(z) for z in coefficients[1:]])
    ft=np.exp(-c*np.outer(t,n))@a
    return np.sum(w*(T-1)/2*ft*(t**(s-1)+epsilon*t**(weight-1-s)))


def plot_lines(x, series: dict, title: str, xlabel: str, ylabel: str,
               logx: bool = False, logy: bool = False, markers: bool = False):
    """単一軸の共通描画。色・スタイルは Matplotlib の既定に任せる。"""
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,4.6))
    for label,y in series.items():
        ax.plot(x,y,marker="o" if markers else None,label=label)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_real_curve(E: EllipticCurve, xmin=-6, xmax=13, points: dict | None=None,
                    line: tuple[float,float] | None=None, title="Real locus of an elliptic curve"):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,4.8))
    x=np.linspace(xmin,xmax,2400)
    a1,a2,a3,a4,a6=map(float,E.ainvs)
    d=(a1*x+a3)**2+4*(x**3+a2*x*x+a4*x+a6)
    root=np.sqrt(np.maximum(d,0));root[d<0]=np.nan
    ax.plot(x,(-(a1*x+a3)+root)/2,label="upper branch")
    ax.plot(x,(-(a1*x+a3)-root)/2,label="lower branch")
    if line:
        ax.plot(x,line[0]*x+line[1],linestyle="--",label="secant / tangent")
    if points:
        for label,P in points.items():
            if P is not None:
                xp,yp=map(float,P)
                ax.scatter([xp],[yp],s=35)
                ax.annotate(label,(xp,yp),xytext=(6,7),textcoords="offset points")
    ax.set(title=title,xlabel="x",ylabel="y")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    return fig


def plot_lattice(tau: complex = .4+1.1j, radius: int = 3):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,5.2))
    lattice=np.r_[0,lattice_vectors(tau,radius)]
    ax.scatter(lattice.real,lattice.imag,s=18,label="lattice points")
    v=np.array([0,1,1+tau,tau,0])
    ax.plot(v.real,v.imag,linewidth=2,label="fundamental parallelogram")
    ax.set_aspect("equal");ax.set(title="Lattice <tau, 1>",xlabel="Re z",ylabel="Im z")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();return fig


def plot_fundamental(path: Sequence[complex]):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,5.2))
    arc=np.linspace(np.pi/3,2*np.pi/3,300)
    ax.plot(np.cos(arc),np.sin(arc),label="|tau| = 1")
    top=max(3,max(z.imag for z in path)+.4)
    ax.plot([-.5,-.5],[np.sqrt(3)/2,top],linestyle="--")
    ax.plot([.5,.5],[np.sqrt(3)/2,top],linestyle="--")
    ax.plot([z.real for z in path],[z.imag for z in path],marker="o",label="S,T reduction path")
    for i,z in enumerate(path):
        ax.annotate(str(i),(z.real,z.imag),xytext=(6,6),textcoords="offset points")
    ax.set(xlabel="Re tau",ylabel="Im tau",title="Fundamental domain of SL2(Z)",ylim=(0,top))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();return fig


def plot_hasse(data: CurveData, prime_bound: int = 1000):
    import matplotlib.pyplot as plt
    primes=[int(p) for p in sp.primerange(2,prime_bound+1) if p not in dict(data.bad_ap)]
    traces=[data.ap(p) for p in primes]
    fig,ax=plt.subplots(figsize=(8.6,4.6))
    ax.scatter(primes,traces,s=14,label="a_p = p + 1 - #E(F_p)")
    xx=np.array(primes)
    ax.plot(xx,2*np.sqrt(xx),linestyle="--",label="Hasse bounds")
    ax.plot(xx,-2*np.sqrt(xx),linestyle="--")
    ax.set(title=f"Frobenius traces: {data.name}",xlabel="prime p",ylabel="a_p")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();return fig


def plot_q_decay():
    y=np.linspace(.01,2,300)
    return plot_lines(y,{"|q| = exp(-2 pi y)":np.exp(-2*np.pi*y)},
                      "Upper half-plane to the q-disk","Im tau","|q|")



# Mazur の15種類を実現する具体的な係数と生成点。位数は厳密演算で検査する。
TORSION_EXAMPLES = [
    ("trivial",(0,0,0,0,-2),[],1),
    ("Z/2",(0,0,0,0,8),[(-2,0)],2),
    ("Z/3",(0,0,0,0,4),[(0,2)],3),
    ("Z/4",(0,0,0,4,0),[(2,4)],4),
    ("Z/5",(0,-1,-1,0,0),[(0,1)],5),
    ("Z/6",(0,0,0,0,1),[(2,3)],6),
    ("Z/7",(0,0,0,-43,166),[(3,8)],7),
    ("Z/8",(7,0,0,16,0),[(-2,10)],8),
    ("Z/9",(1,-1,1,-14,29),[(3,1)],9),
    ("Z/10",(1,0,0,-45,81),[(0,9)],10),
    ("Z/12",(43,-210,-210,0,0),[(0,210)],12),
    ("Z/2 x Z/2",(0,0,0,-4,0),[(2,0),(0,0)],4),
    ("Z/4 x Z/2",(0,2,0,-3,0),[(3,6),(0,0)],8),
    ("Z/6 x Z/2",(5,-3,-6,0,0),[(-3,18),(2,-2)],12),
    ("Z/8 x Z/2",(17,-60,-120,0,0),[(30,-90),(-40,400)],16),
]


def torsion_order_candidates(E: EllipticCurve, subgroup_order: int,
                             prime_bound: int = 43) -> list[int]:
    """Mazur の位数候補を、良い還元の素数-to-p 単射で厳密に絞る。

    subgroup_order は実際に生成して確認した有限部分群の位数であること。
    これは任意の曲線の全 torsion を構成する関数ではない。
    戻り値が [subgroup_order] なら与えた部分群の完全性が従う。
    """
    if subgroup_order < 1 or prime_bound < 3:
        raise ValueError("正の部分群位数と prime_bound >= 3 が必要です。")
    possible = sorted(set(range(1, 11)) | {12, 16})
    possible = [t for t in possible if t % subgroup_order == 0]
    for p in sp.primerange(2, prime_bound + 1):
        p = int(p)
        if not E.smooth_mod_p(p):
            continue
        count = E.count_mod_p(p)
        survivors = []
        for t in possible:
            prime_to_p = t
            while prime_to_p % p == 0:
                prime_to_p //= p
            if count % prime_to_p == 0:
                survivors.append(t)
        possible = survivors
    return possible


def verify_torsion_examples() -> list[dict]:
    """15例の生成元・生成部分群と、全 torsion の位数を厳密に検査。"""
    rows=[]
    for label,coeff,gens,expected in TORSION_EXAMPLES:
        E=EllipticCurve(*coeff)
        points=[E.point(*P) for P in gens]
        order=len(torsion_generated(E,points))
        possible = torsion_order_candidates(E, order)
        if order != expected or possible != [order]:
            raise AssertionError(f"{label} の torsion の完全性を確認できません。候補={possible}")
        rows.append(dict(group=label,ainvs=str(coeff),generators=str(gens),
                         generator_orders=[E.order_up_to(P) for P in points],
                         generated_order=order,total_order_candidates=str(possible),
                         completeness="Mazur + 良い還元の単射で確定"))
    return rows


def tate_normal_form(b,c) -> EllipticCurve:
    b,c=rational(b),rational(c)
    return EllipticCurve(1-c,-b,-b,0,0)


def projective_points_fp(p: int, dimension: int = 2) -> set[tuple[int,...]]:
    if not sp.isprime(p) or dimension < 1 or p**(dimension+1)>1_000_000:
        raise ValueError("小さな素数と次元に限定した総当たりです。")
    result=set()
    for v in product(range(p),repeat=dimension+1):
        nz=next((x for x in v if x),None)
        if nz is not None:
            inverse=pow(nz,-1,p)
            result.add(tuple(x*inverse%p for x in v))
    return result


if __name__ == "__main__":
    print(run_self_tests())
