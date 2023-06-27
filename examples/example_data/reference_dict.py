# Default equipment dictionary is based on
# https://www.osti.gov/servlets/purl/1872704
# TODO: only currently takes single words, not phrases: 'string inverter', 'met station', etc.
EQUIPMENT_DICT = {'combiner': ['combiner', 'comb', 'cb'],
                  'battery': ['battery', 'bess', ],  # this should be energy storage, if we could handle n-grams
                  'inverter': ['inverter', 'invert', 'inv', ],
                  'met': ['met'],  # this should be met station, if we could handle n-grams
                  'meter': ['meter'],
                  'module': ['module', 'mod'],
                  'recloser': ['recloser', 'reclose'],
                  'relay': ['relay'],
                  'substation': ['substation'],
                  'switchgear': ['switchgear', 'switch'],
                  'tracker': ['tracker'],
                  'transformer': ['transformer', 'xfmr'],
                  'wiring': ['wiring', 'wire', 'wires']
                  }

PV_TERMS_DICT = {'communication': ['comm',],
                 'energy': ['energy', 'kwh', 'mwh',],
                 'grid': ['grid', 'curtailment', 'curtail', 'poi',],
                 'outage': ['outage', 'offline',],
                 'solar': ['solar', 'pv', 'photovoltaic',],
                 'system': ['system', 'site', 'farm', 'project',],
                 'make_model': ['sma',], # TODO: use the equipment database for this
                 'corrective_maintence': ['cm',],
                 'preventative_maintence': ['pm',]
                 }
