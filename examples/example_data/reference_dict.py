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