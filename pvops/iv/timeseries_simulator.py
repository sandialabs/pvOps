import simulator


class IVTimeseriesGenerator(simulator.Simulator):

    def __init__(self, **iv_sim_kwargs):
        """Simulate a PV System across time.

        Parameters
        ----------
        iv_sim_kwargs :
            Optional, `simulator.Simulator` inputs
        """
        super().__init__(**iv_sim_kwargs)

    def generate(self, env_df, failure_settings):
        """Simulate a PV system

        Parameters
        ----------
        env_df : dataframe
            DataFrame containing irradiance ("E") and temperature ("T") columns
        degradation_settings : dict
            Define timeseries degradation patterns
        """
        env_df = env_df.copy()
        failure_settings.get('')
