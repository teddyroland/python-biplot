import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

## import data

dat = pd.read_csv('data.csv', index_col=0)
# if no row or column titles in your csv, pass 'header=None' into read_csv
# and delete 'index_col=0' -- but your biplot will be better with row/col names



## perform PCA

pca = PCA(n_components=len(dat.columns))
# defaults # PCs to # of columns in imported data, but can be set to any
# integer less than or equal to that value

pca.fit(dat)



## project data into PC space

xaxis = pca.components_[0] # see pca$rotation in R
yaxis = pca.components_[1] # change 0,1 for other principle components

xs = pca.transform(dat)[:,0] # see pca$x in R
ys = pca.transform(dat)[:,1] # change 0,1 for other principle components



## visualize projections
    
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them

for i in range(len(xaxis)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xaxis[i]*max(xs), yaxis[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xaxis[i]*max(xs)*1.2, yaxis[i]*max(ys)*1.2,
             list(dat.columns.values)[i], color='r')

for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(dat.index)[i], color='b')

plt.show()
