```
conda create -n clrs python=3.10

pip install -e .

Feedback(
    features=Features(
    inputs=(DataPoint(name="pos",        location=node,  type=scalar,    data=Array(4, 16)), DataPoint(name="s", location=node,  type=mask_one,  data=Array(4, 16)), DataPoint(name="A", location=edge,  type=scalar,    data=Array(4, 16, 16)), DataPoint(name="adj",       location=edge,  type=mask,      data=Array(4, 16, 16))), 
    hints=(DataPoint(name="reach_h",       location=node,  type=mask,      data=Array(10, 4, 16)), DataPoint(name="pi_h",  location=node,  type=pointer,   data=Array(10, 4, 16))), lengths=array([3., 1., 3., 3.])), 
    
    outputs=[DataPoint(name="pi",    location=node,  type=pointer,   data=Array(4, 16))])

```

### Haiku

Transforms a function using Haiku modules into a pair of pure functions.

For a function ``out = f(*a, **k)`` this function returns a pair of two pure
functions that call ``f(*a, **k)`` explicitly collecting and injecting
parameter values::

    params = init(rng, *a, **k)
    out = apply(params, rng, *a, **k)

Note that the ``rng`` argument is typically not required for ``apply`` and
passing ``None`` is accepted.

The first thing to do is to define a :class:`Module`. A module encapsulates
some parameters and a computation on those parameters:

>>> class MyModule(hk.Module):
...   def __call__(self, x):
...     w = hk.get_parameter("w", [], init=jnp.zeros)
...     return x + w

Next, define some function that creates and applies modules. We use
:func:`transform` to transform that function into a pair of functions that
allow us to lift all the parameters out of the function (``f.init``) and
apply the function with a given set of parameters (``f.apply``):

>>> def f(x):
...   a = MyModule()
...   b = MyModule()
...   return a(x) + b(x)
>>> f = hk.transform(f)

To get the initial state of the module call ``init`` with an example input:

>>> params = f.init(None, 1)
>>> params
{'my_module': {'w': ...Array(0., dtype=float32)},
'my_module_1': {'w': ...Array(0., dtype=float32)}}

You can then apply the function with the given parameters by calling
``apply`` (note that since we don't use Haiku's random number APIs to apply
our network we pass ``None`` as an RNG key):

>>> print(f.apply(params, None, 1))
2.0

It is expected that your program will at some point produce updated parameters
and you will want to re-apply ``apply``. You can do this by calling ``apply``
with different parameters:

>>> new_params = {"my_module": {"w": jnp.array(2.)},
...               "my_module_1": {"w": jnp.array(3.)}}
>>> print(f.apply(new_params, None, 2))
9.0

If your transformed function needs to maintain internal state (e.g. moving
averages in batch norm) then see :func:`transform_with_state`.

