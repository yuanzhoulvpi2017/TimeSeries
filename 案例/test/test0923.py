from scipy import signal

from statsmodels.tsa.stattools import yule_walker

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima_process import ArmaProcess

import numpy as np

data = np.random.randint(low=10, high=100, size=(10,2))


class ArmaProcess(object):
    r"""
    Theoretical properties of an ARMA process for specified lag-polynomials.

    Parameters
    ----------
    ar : array_like
        Coefficient for autoregressive lag polynomial, including zero lag.
        Must be entered using the signs from the lag polynomial representation.
        See the notes for more information about the sign.
    ma : array_like
        Coefficient for moving-average lag polynomial, including zero lag.
    nobs : int, optional
        Length of simulated time series. Used, for example, if a sample is
        generated. See example.

    Notes
    -----
    Both the AR and MA components must include the coefficient on the
    zero-lag. In almost all cases these values should be 1. Further, due to
    using the lag-polynomial representation, the AR parameters should
    have the opposite sign of what one would write in the ARMA representation.
    See the examples below.

    The ARMA(p,q) process is described by

    .. math::

        y_{t}=\phi_{1}y_{t-1}+\ldots+\phi_{p}y_{t-p}+\theta_{1}\epsilon_{t-1}
               +\ldots+\theta_{q}\epsilon_{t-q}+\epsilon_{t}

    and the parameterization used in this function uses the lag-polynomial
    representation,

    .. math::

        \left(1-\phi_{1}L-\ldots-\phi_{p}L^{p}\right)y_{t} =
            \left(1+\theta_{1}L+\ldots+\theta_{q}L^{q}\right)\epsilon_{t}

    Examples
    --------
    ARMA(2,2) with AR coefficients 0.75 and -0.25, and MA coefficients 0.65 and 0.35

    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> arma_process = sm.tsa.ArmaProcess(ar, ma)
    >>> arma_process.isstationary
    True
    >>> arma_process.isinvertible
    True
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    >>> y = arma_process.generate_sample(250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])

    The same ARMA(2,2) Using the from_coeffs class method

    >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(arparams, maparams)
    >>> arma_process.arroots
    array([1.5-1.32287566j, 1.5+1.32287566j])
    """

    # TODO: Check unit root behavior
    def __init__(self, ar=None, ma=None, nobs=100):
        if ar is None:
            ar = np.array([1.])
        if ma is None:
            ma = np.array([1.])
        self.ar = array_like(ar, 'ar')
        self.ma = array_like(ma, 'ma')
        self.arcoefs = -self.ar[1:]
        self.macoefs = self.ma[1:]
        self.arpoly = np.polynomial.Polynomial(self.ar)
        self.mapoly = np.polynomial.Polynomial(self.ma)
        self.nobs = nobs

    @classmethod
    def from_coeffs(cls, arcoefs=None, macoefs=None, nobs=100):
        """
        Create ArmaProcess from an ARMA representation.

        Parameters
        ----------
        arcoefs : array_like
            Coefficient for autoregressive lag polynomial, not including zero
            lag. The sign is inverted to conform to the usual time series
            representation of an ARMA process in statistics. See the class
            docstring for more information.
        macoefs : array_like
            Coefficient for moving-average lag polynomial, excluding zero lag.
        nobs : int, optional
            Length of simulated time series. Used, for example, if a sample
            is generated.

        Returns
        -------
        ArmaProcess
            Class instance initialized with arcoefs and macoefs.

        Examples
        --------
        >>> arparams = [.75, -.25]
        >>> maparams = [.65, .35]
        >>> arma_process = sm.tsa.ArmaProcess.from_coeffs(ar, ma)
        >>> arma_process.isstationary
        True
        >>> arma_process.isinvertible
        True
        """
        arcoefs = [] if arcoefs is None else arcoefs
        macoefs = [] if macoefs is None else macoefs
        return cls(np.r_[1, -np.asarray(arcoefs)],
                   np.r_[1, np.asarray(macoefs)],
                   nobs=nobs)

    @classmethod
    def from_estimation(cls, model_results, nobs=None):
        """
        Create an ArmaProcess from the results of an ARMA estimation.

        Parameters
        ----------
        model_results : ARMAResults instance
            A fitted model.
        nobs : int, optional
            If None, nobs is taken from the results.

        Returns
        -------
        ArmaProcess
            Class instance initialized from model_results.
        """
        arcoefs = model_results.arparams
        macoefs = model_results.maparams
        nobs = nobs or model_results.nobs
        return cls(np.r_[1, -arcoefs], np.r_[1, macoefs], nobs=nobs)

    def __mul__(self, oth):
        if isinstance(oth, self.__class__):
            ar = (self.arpoly * oth.arpoly).coef
            ma = (self.mapoly * oth.mapoly).coef
        else:
            try:
                aroth, maoth = oth
                arpolyoth = np.polynomial.Polynomial(aroth)
                mapolyoth = np.polynomial.Polynomial(maoth)
                ar = (self.arpoly * arpolyoth).coef
                ma = (self.mapoly * mapolyoth).coef
            except:
                raise TypeError('Other type is not a valid type')
        return self.__class__(ar, ma, nobs=self.nobs)

    def __repr__(self):
        msg = 'ArmaProcess({0}, {1}, nobs={2}) at {3}'
        return msg.format(self.ar.tolist(), self.ma.tolist(),
                          self.nobs, hex(id(self)))

    def __str__(self):
        return 'ArmaProcess\nAR: {0}\nMA: {1}'.format(self.ar.tolist(),
                                                      self.ma.tolist())

    @Appender(remove_parameters(arma_acovf.__doc__, ['ar', 'ma', 'sigma2']))
    def acovf(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_acovf(self.ar, self.ma, nobs=nobs)

    @Appender(remove_parameters(arma_acf.__doc__, ['ar', 'ma']))
    def acf(self, lags=None):
        lags = lags or self.nobs
        return arma_acf(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma_pacf.__doc__, ['ar', 'ma']))
    def pacf(self, lags=None):
        lags = lags or self.nobs
        return arma_pacf(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma_periodogram.__doc__, ['ar', 'ma', 'worN',
                                                           'whole']))
    def periodogram(self, nobs=None):
        nobs = nobs or self.nobs
        return arma_periodogram(self.ar, self.ma, worN=nobs)

    @Appender(remove_parameters(arma_impulse_response.__doc__, ['ar', 'ma']))
    def impulse_response(self, leads=None):
        leads = leads or self.nobs
        return arma_impulse_response(self.ar, self.ma, leads=leads)

    @Appender(remove_parameters(arma2ma.__doc__, ['ar', 'ma']))
    def arma2ma(self, lags=None):
        lags = lags or self.lags
        return arma2ma(self.ar, self.ma, lags=lags)

    @Appender(remove_parameters(arma2ar.__doc__, ['ar', 'ma']))
    def arma2ar(self, lags=None):
        lags = lags or self.lags
        return arma2ar(self.ar, self.ma, lags=lags)

    @property
    def arroots(self):
        """Roots of autoregressive lag-polynomial"""
        return self.arpoly.roots()

    @property
    def maroots(self):
        """Roots of moving average lag-polynomial"""
        return self.mapoly.roots()

    @property
    def isstationary(self):
        """
        Arma process is stationary if AR roots are outside unit circle.

        Returns
        -------
        bool
             True if autoregressive roots are outside unit circle.
        """
        if np.all(np.abs(self.arroots) > 1.0):
            return True
        else:
            return False

    @property
    def isinvertible(self):
        """
        Arma process is invertible if MA roots are outside unit circle.

        Returns
        -------
        bool
             True if moving average roots are outside unit circle.
        """
        if np.all(np.abs(self.maroots) > 1):
            return True
        else:
            return False

    def invertroots(self, retnew=False):
        """
        Make MA polynomial invertible by inverting roots inside unit circle.

        Parameters
        ----------
        retnew : bool
            If False (default), then return the lag-polynomial as array.
            If True, then return a new instance with invertible MA-polynomial.

        Returns
        -------
        manew : ndarray
           A new invertible MA lag-polynomial, returned if retnew is false.
        wasinvertible : bool
           True if the MA lag-polynomial was already invertible, returned if
           retnew is false.
        armaprocess : new instance of class
           If retnew is true, then return a new instance with invertible
           MA-polynomial.
        """
        # TODO: variable returns like this?
        pr = self.maroots
        mainv = self.ma
        invertible = self.isinvertible
        if not invertible:
            pr[np.abs(pr) < 1] = 1. / pr[np.abs(pr) < 1]
            pnew = np.polynomial.Polynomial.fromroots(pr)
            mainv = pnew.coef / pnew.coef[0]

        if retnew:
            return self.__class__(self.ar, mainv, nobs=self.nobs)
        else:
            return mainv, invertible

    @Appender(str(_generate_sample_doc))
    def generate_sample(self, nsample=100, scale=1., distrvs=None, axis=0,
                        burnin=0):
        return arma_generate_sample(self.ar, self.ma, nsample, scale, distrvs,
                                    axis=axis, burnin=burnin)




def arma_generate_sample(ar, ma, nsample, scale=1, distrvs=None,
                         axis=0, burnin=0):
    """
    Simulate data from an ARMA.

    Parameters
    ----------
    ar : array_like
        The coefficient for autoregressive lag polynomial, including zero lag.
    ma : array_like
        The coefficient for moving-average lag polynomial, including zero lag.
    nsample : int or tuple of ints
        If nsample is an integer, then this creates a 1d timeseries of
        length size. If nsample is a tuple, creates a len(nsample)
        dimensional time series where time is indexed along the input
        variable ``axis``. All series are unless ``distrvs`` generates
        dependent data.
    scale : float
        The standard deviation of noise.
    distrvs : function, random number generator
        A function that generates the random numbers, and takes ``size``
        as argument. The default is np.random.standard_normal.
    axis : int
        See nsample for details.
    burnin : int
        Number of observation at the beginning of the sample to drop.
        Used to reduce dependence on initial values.

    Returns
    -------
    ndarray
        Random sample(s) from an ARMA process.

    Notes
    -----
    As mentioned above, both the AR and MA components should include the
    coefficient on the zero-lag. This is typically 1. Further, due to the
    conventions used in signal processing used in signal.lfilter vs.
    conventions in statistics for ARMA processes, the AR parameters should
    have the opposite sign of what you might expect. See the examples below.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(12345)
    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> ar = np.r_[1, -arparams] # add zero-lag and negate
    >>> ma = np.r_[1, maparams] # add zero-lag
    >>> y = sm.tsa.arma_generate_sample(ar, ma, 250)
    >>> model = sm.tsa.ARMA(y, (2, 2)).fit(trend='nc', disp=0)
    >>> model.params
    array([ 0.79044189, -0.23140636,  0.70072904,  0.40608028])
    """
    distrvs = np.random.standard_normal if distrvs is None else distrvs
    if np.ndim(nsample) == 0:
        nsample = [nsample]
    if burnin:
        # handle burin time for nd arrays
        # maybe there is a better trick in scipy.fft code
        newsize = list(nsample)
        newsize[axis] += burnin
        newsize = tuple(newsize)
        fslice = [slice(None)] * len(newsize)
        fslice[axis] = slice(burnin, None, None)
        fslice = tuple(fslice)
    else:
        newsize = tuple(nsample)
        fslice = tuple([slice(None)] * np.ndim(newsize))
    eta = scale * distrvs(size=newsize)
    return signal.lfilter(ma, ar, eta, axis=axis)[fslice]
