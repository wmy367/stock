
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels.api as sm
import numpy as np

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]
nobs = 250
np.random.seed(2014)
y = arma_generate_sample(arparams, maparams, nobs)
res = sm.tsa.arma_order_select_ic(y, ic=["aic", "bic"], trend="nc")
print(res.aic_min_order)
print(res.bic_min_order)