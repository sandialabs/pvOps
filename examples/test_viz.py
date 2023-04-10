import pandas as pd
import matplotlib.pyplot as plt
from pvops.text.visualize import visualize_attribute_connectivity

df = pd.read_csv('om_df.csv')

om_col_dict = {
    'attribute1_col': 'Asset',
    'attribute2_col': 'ImpactLevel'
}

visualize_attribute_connectivity(
    df, 
    om_col_dict,
    # Optional
    graph_aargs = {
        'with_labels':True,
        'font_weight':'bold',
        'node_size':500,
        'font_size':10
    }
)
plt.show()

foo