"""WPE classes and methods."""


class WPE(object):
    """WPE Class."""

    def __init__(
        self, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode="full"
    ):
        """Initialize WPE."""
        self.taps = taps
        self.delay = delay
        self.iterations = iterations
        self.psd_context = psd_context
        self.statistics_mode = statistics_mode

    def __repr__(self):
        """Return string representation of class."""
        return (
            "{name}(taps={taps}, delay={delay}"
            "iterations={iterations}, psd_context={psd_context}, "
            "statistics_mode={statistics_mode})".format(
                name=self.__class__.__name__,
                taps=self.taps,
                delay=self.delay,
                iterations=self.iterations,
                psd_context=self.psd_context,
                statistics_mode=self.statistics_mode,
            )
        )

    def __call__(self, xs):
        """Return enhanced.

        :param np.ndarray xs: (Time, Channel, Frequency)
        :return: enhanced_xs
        :rtype: np.ndarray

        """
        from nara_wpe.wpe import wpe

        # nara_wpe.wpe: (F, C, T)
        xs = wpe(
            xs.transpose((2, 1, 0)),
            taps=self.taps,
            delay=self.delay,
            iterations=self.iterations,
            psd_context=self.psd_context,
            statistics_mode=self.statistics_mode,
        )
        return xs.transpose(2, 1, 0)
