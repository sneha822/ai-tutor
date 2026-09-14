# Calculus: Derivatives

Sample course notes. Replace with your own materials; any .md, .txt or .pdf in this folder is indexed.

## 1. What a derivative is

The derivative of a function measures how fast its output changes as its input changes. Geometrically, it is the slope of the tangent line to the graph at a point.

Formally, the derivative of $f$ at $x$ is the limit of the difference quotient:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

If this limit exists, $f$ is differentiable at $x$. Differentiable functions are always continuous, but continuous functions are not always differentiable. For example, $|x|$ is continuous at $0$ but has a sharp corner there, so it has no derivative at $0$.

Common notations: $f'(x)$, $\frac{dy}{dx}$, $\frac{d}{dx}f(x)$, and $Df$.

## 2. Basic rules

- Constant rule: $\frac{d}{dx} c = 0$
- Power rule: $\frac{d}{dx} x^n = n x^{n-1}$ for any real $n$
- Constant multiple: $\frac{d}{dx}[c f(x)] = c f'(x)$
- Sum rule: $\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)$

Example: $\frac{d}{dx}(3x^4 - 5x + 7) = 12x^3 - 5$.

A common mistake with the power rule is forgetting to subtract one from the exponent, or applying it to $e^x$ or $2^x$, where the variable is in the exponent. The power rule only applies when the variable is in the base.

## 3. Product and quotient rules

Product rule: the derivative of a product is not the product of the derivatives.

$$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$

Example: $\frac{d}{dx}[x^2 \sin x] = 2x \sin x + x^2 \cos x$.

Quotient rule ("low d-high minus high d-low, over low squared"):

$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}$$

## 4. The chain rule

The chain rule differentiates compositions of functions, a function inside another function. Differentiate the outer function, leaving the inside alone, then multiply by the derivative of the inside.

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

In Leibniz notation, with $y = f(u)$ and $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Example 1: $\frac{d}{dx} (3x^2 + 1)^5 = 5(3x^2 + 1)^4 \cdot 6x = 30x(3x^2+1)^4$.

Example 2: $\frac{d}{dx} \sin(x^2) = \cos(x^2) \cdot 2x$.

Example 3: $\frac{d}{dx} e^{4x} = 4e^{4x}$.

The most common chain rule mistake is forgetting to multiply by the derivative of the inner function. A good habit: when you see parentheses, a function of something other than plain $x$, or an exponent containing $x$, ask "what is the inside?"

## 5. Derivatives of common functions

- $\frac{d}{dx} \sin x = \cos x$
- $\frac{d}{dx} \cos x = -\sin x$
- $\frac{d}{dx} \tan x = \sec^2 x$
- $\frac{d}{dx} e^x = e^x$
- $\frac{d}{dx} a^x = a^x \ln a$
- $\frac{d}{dx} \ln x = \frac{1}{x}$ for $x > 0$

## 6. Applications

Tangent lines: the tangent line to $y = f(x)$ at $x = a$ is $y = f(a) + f'(a)(x - a)$.

Rates of change: if $s(t)$ is position, then $v(t) = s'(t)$ is velocity and $a(t) = v'(t) = s''(t)$ is acceleration.

Optimization: local maxima and minima of a differentiable function occur where $f'(x) = 0$ (critical points). The second derivative test: if $f'(c) = 0$ and $f''(c) > 0$, then $f$ has a local minimum at $c$; if $f''(c) < 0$, a local maximum.

Example: a farmer has 100 m of fence for a rectangular pen. Area $A = x(50 - x)$, so $A'(x) = 50 - 2x = 0$ gives $x = 25$. The maximum area is a $25 \times 25$ square, $625\text{ m}^2$.
