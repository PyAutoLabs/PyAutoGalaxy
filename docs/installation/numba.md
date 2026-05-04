(numba)=

# Numba

Numba (<https://numba.pydata.org>) is an optional library which makes **PyAutoGalaxy** run a lot faster, which we strongly
recommend all users have installed.

Certain functionality (pixelized source reconstructions, linear light profiles) is disabled without numba installed
because it will have too slow run-times.

However, some users have experienced difficulties installing numba, meaning they have been unable to try out
**PyAutoGalaxy** and determine if it the right software for them, before committing more time to installing numba
successfully.

For this reason, numba is an optional installation, so that users can easily experiment and learn
the basic API.

If you do not have numba installed, you can do so via pip as follows:

```bash
pip install numba
```
