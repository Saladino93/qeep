import vegas
import jax
#jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_enable_x64', False)
jnp = jax.numpy

@jax.jit
def _jridge(x):
    N = 1000
    x0 = jnp.linspace(0.25, 0.75, N)
    dx2 = 0.0
    for xd in x:
        dx2 += (xd[None,:] - x0[:, None]) ** 2
    return jnp.sum(jnp.exp(-100. * dx2), axis=0) *  (100. / jnp.pi) ** (len(x) / 2.) / N

@vegas.rbatchintegrand
def ridge(x):
    return _jridge(jnp.array(x))

integ = vegas.Integrator(4 * [[0, 1]], gpu_pad=True)

integ(ridge, nitn=10, neval=2e5)
result = integ(ridge, nitn=10, neval=2e5, adapt=False)
print('result = %s   Q = %.2f' % (result, result.Q))