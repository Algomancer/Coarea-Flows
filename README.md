# Coarea Normalizing Flow

A minimal PyTorch implementation of a **graph-parameterized coarea normalizing flow**.

The core object is a flow layer that turns a point `x` into two coordinates:

```text
z = (t, u)
```

where `t` is the value of a learned scalar function, and `u` is the coordinate of the point after sliding it down to a fixed reference level set.

Most normalizing flows build invertible maps from coupling blocks, autoregressive transforms, splines, or residual dynamics. This one builds an invertible map from the geometry of **level sets**.

## The main idea

Pick a scalar function

```text
f : R^d -> R
```

and use it as a learned height function. Every point `x` gets assigned a height

```text
t = f(x)
```

The sets where `f(x) = constant` are the level sets of `f`, like contour lines on a map.

The layer represents `x` by:

```text
t = which level set is x on?
u = where does x land on the base level set f = a?
```

So the layer is not just bending space arbitrarily. It is building a coordinate system from a learned foliation of the space.

## Graph parameterization

The implementation uses the special form

```text
x = (x_1, x_rest)

f(x) = x_1 + g(x_rest)
```

where `g` is a small neural network.

This is the key design choice.

Because `f` is written as a graph over the remaining coordinates, the base level set

```text
f(x) = a
```

has an explicit global chart:

```text
phi(u) = (a - g(u), u)
```

That means if we know `u`, we can immediately write down the point on the base level set.

It also means the gradient never vanishes:

```text
|grad f|^2 = 1 + |grad g|^2 >= 1
```

So there are no critical points where the construction gets stuck.

## The coarea coordinate map

The layer uses the vector field

```text
v(x) = grad f(x) / |grad f(x)|^2
```

This field has one important property:

```text
d/ds f(x(s)) = grad f(x(s)) · v(x(s)) = 1
```

So following `v` moves through the level sets of `f` at unit speed.

That gives a clean inverse pair.

### Forward: data space to latent space

Given `x`:

```text
t = f(x)
```

Then follow the flow from level `t` down to the base level `a`:

```text
x_a = flow(x, from level t, to level a)
```

The point `x_a` lies on the base level set. Since the base level is parameterized by `u`, we keep its remaining coordinates:

```text
u = (x_a)_rest
```

The output is

```text
z = (t, u)
```

In code:

```python
t = x[:, 0] + self.g(x[:, 1:])
x_a, ldj = self.integrate(x, t, torch.full_like(t, self.a))
z = torch.cat([t.unsqueeze(-1), x_a[:, 1:]], dim=-1)
```

### Inverse: latent space to data space

Given `z = (t, u)`, start on the base level set:

```text
q = phi(u) = (a - g(u), u)
```

Then follow the same vector field from level `a` up to level `t`:

```text
x = flow(q, from level a, to level t)
```

In code:

```python
t, u = z[:, 0], z[:, 1:]
q1 = self.a - self.g(u)
q = torch.cat([q1.unsqueeze(-1), u], dim=-1)
x, ldj = self.integrate(q, torch.full_like(t, self.a), t)
```

This is the coarea bijection:

```text
(t, u) <-> x
```

The coarea layer gives both.

The invertible map comes from the level-set coordinate system:

```text
x -> (f(x), base-coordinate(x))
```

The log-volume change comes from the divergence of the vector field along the trajectory. In this implementation, the field and its divergence are written in closed form for the one-hidden-layer `g`, so the layer can accumulate the log-determinant directly while moving between level sets.

The density formula is the usual flow formula:

```text
log p_X(x) = log p_Z(z) + log |det dz/dx|
```

where `p_Z` is a standard normal.

## What `g` learns

The network `g` bends the level sets.

If `g = 0`, then

```text
f(x) = x_1
```

and the level sets are flat coordinate slices.

As `g` trains, those flat slices become curved. The layer learns a geometry in which data points are easier to express as `(t, u)` coordinates.

The implementation initializes the final weights of `g` to zero, so the layer starts close to the identity. Training then gradually bends the level sets instead of beginning with a highly distorted map.

## Summary

This flow learns a scalar function `f`, uses its level sets as a coordinate system, and maps each point to `(t, u)`: its level value and its coordinate on a fixed base level. The graph form `f(x) = x_1 + g(x_rest)` makes the base chart explicit and prevents critical points. The normalized-gradient field moves exactly from one level set to another, giving a clean inverse. Stacking these layers with rotations and affine normalization produces a trainable normalizing flow with exact likelihood accounting up to the numerical accuracy of the level-set transport.

## Citation

```bibtex
@software{algomancer,
  author = {{@algomancer}},
  title = {Coarea Normalizing Flow},
  year = {2026}
}
```
