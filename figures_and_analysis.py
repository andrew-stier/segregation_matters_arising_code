import pickle
from os.path import join
from pathlib import Path

import numpy as np
import pandas
import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm, binned_statistic
from statsmodels.api import OLS, add_constant, Logit, categorical
import scipy
from statsmodels.stats.multitest import multipletests
from met_brewer.palettes import met_brew
from scipy.stats import pearsonr, spearmanr
from concurrent.futures import ProcessPoolExecutor

exposures = pandas.read_csv('exposures.csv', sep=' ')
exposures["#Exposures"] = exposures['#Exposures'].str.replace(',', '').astype(float).values

cbsas_delineation = pandas.read_csv('delineation_2020.csv', skiprows=0)
cbsas_delineation['FIPS State Code'] = cbsas_delineation['FIPS State Code'].astype(str).map(
    lambda x: x if len(x) == 2 else '0' + x)
cbsas_delineation['FIPS County Code'] = cbsas_delineation['FIPS County Code'].astype(str).map(
    lambda x: x if len(x) == 3 else ('0' + x) if len(x) == 2 else '00' + x)
cbsas_delineation['county'] = cbsas_delineation['FIPS State Code'].astype(str) + cbsas_delineation[
    'FIPS County Code'].astype(str)
cbsas_delineation['CBSA Code'] = cbsas_delineation['CBSA Code'].astype(int)

exposures['cbsa_code'] = list(map(lambda x: x[0] if ~numpy.isnan(x[0]) else x[1],
                                  zip([x[0] if len(x) > 0 else numpy.nan for x in [cbsas_delineation[cbsas_delineation[
                                      'CBSA Title'].str.split(',').str[0].str.lower().str.replace(' ', '').str.contains(
                                      x)]['CBSA Code'].values for x in exposures['MSA'].str.split(',').str[
                                                                                       0].str.lower().str.split(
                                      '-').str[0]]], [x[0] if len(x) > 0 else numpy.nan for x in [cbsas_delineation[
                                                                                                      cbsas_delineation[
                                                                                                          'CBSA Title'].str.split(
                                                                                                          ',').str[
                                                                                                          0].str.lower().str.replace(
                                                                                                          ' ',
                                                                                                          '').str.contains(
                                                                                                          x)][
                                                                                                      'CBSA Code'].values
                                                                                                  for x in exposures[
                                                                                                      'MSA'].str.split(
                                          ',').str[0].str.lower().str.split('-').str[-1]]])))


##################################
data_path = 'cbsa_data/'

data = pandas.read_csv(join(data_path, "CTScalingFiles/ct.distr.csv"))
idmappings = pandas.read_excel(join(data_path, "CTScalingFiles/cbsaidmappings.xls")).set_index("CBSA Code")
#
income_lists = {}
# maps MSA id to census tract population
pops = {}
fullpops = {}
cts = set()
i = 0
# populations of whole cities (DONE: Cross-check with other data)
city_pops = {}

metro_ids = set()
micro_ids = set()
for _, ct in data.iterrows():
    i += 1
    msa = ct['msa']
    ct_id = ct["ct"]

    # NOTE on data: "meandollars is total ct income / # workers... "
    ct_inc = ct["ct.meandollars"]
    pop = ct["ct.totalworkers"]
    fullpop = ct["ct.population"]
    if (ct["ct"], msa) in cts:
        print("WARNING! Found Double census tract id: " + str(ct["ct"]))
    cts.add((ct["ct"], msa))

    if msa not in city_pops:
        city_pops[msa] = ct["pop.sum"]

    if msa not in income_lists or msa not in pops:
        income_lists[msa] = []
        pops[msa] = []
        fullpops[msa] = []
    income_lists[msa].append(ct_inc)
    pops[msa].append(pop)
    fullpops[msa].append(fullpop)
    msa = str(msa)
    metro_micro_lbl = idmappings["Metropolitan/Micropolitan Statistical Area"][msa]
    if not isinstance(metro_micro_lbl, str):
        metro_micro_lbl = list(metro_micro_lbl)[0]
    metro = "Metro" in metro_micro_lbl
    if metro:
        metro_ids.add(msa)
    else:
        micro_ids.add(msa)

n_cts = {}
for _, ct in data.iterrows():
    msa = ct['msa']
    if True:  # str(msa) in metro_ids:
        ct_id = ct["ct"]
        if msa not in n_cts:
            n_cts[msa] = 0
        n_cts[msa] += 1

n_cts_list = []
for msa in income_lists:
    if str(msa) in metro_ids:
        n_cts_list.append(n_cts[msa])

kde_distr = {}
kde_distr_small = {}
kde_pdfs = {}
x_inc_all = []
y_pop_all = []
kde = {}


def wide_silverman(kde_inst):
    n = kde_inst.n
    d = kde_inst.d
    silverman = (n * (d + 2) / 4.) ** (-1. / (d + 4))
    scale = 2.
    return silverman * scale


kde_factor = 1 / numpy.linspace(.0001, .1, 100)[numpy.argmin([numpy.abs((exposures['#Exposures'].astype(
    float).values / (t * exposures['Pop.Size'].str.replace(',', '').astype(float).values)).mean() - 363) for t in
                                                              numpy.linspace(.0001, .1, 100)])]


def draw_kde_one_city(i):
    if numpy.isnan(i):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
    # copied everything in for loop
    # msa = CBSA Code for metro area
    try:
        msa = int(exposures['cbsa_code'].values[i])
        income_list = income_lists[msa]
        x = np.linspace(.2 * min(income_list), 1.5 * max(income_list), 500)

        income_list = np.array(income_list)
        pop = np.array(fullpops[msa]) / sum(fullpops[msa])

        kernel = scipy.stats.gaussian_kde(income_list, weights=pop, bw_method=wide_silverman)

        xmin, xmax = x[numpy.argwhere(
            (kernel.pdf(x) * exposures['Pop.Size'].str.replace(',', '').astype(float).values[i]) > 1).flatten()[
            [0, -1]]]

        x_sample = numpy.random.choice(x, size=250, p=kernel.pdf(x) / kernel.pdf(x).sum(), replace=True)
        thresholds = numpy.exp(numpy.linspace(numpy.log(20000), numpy.log(300000), 150))
        cors = [pearsonr(x_sample, [numpy.random.choice(x[(x > (p - t)) & (x < (p + t))], size=250,
                                                        p=kernel.pdf(x[(x > (p - t)) & (x < (p + t))]) / kernel.pdf(
                                                            x[(x > (p - t)) & (x < (p + t))]).sum(),
                                                        replace=True).mean()
                                    for p in x_sample]).statistic for t in
                thresholds]
        cor_arg = numpy.argmin(numpy.abs(numpy.array(cors) - exposures['ExposureSegregation'][i]))
        threshold = thresholds[cor_arg]

        pop = exposures['Pop.Size'].str.replace(',', '').astype(float).values[i]
        exposure = exposures['#Exposures'].values[i]
        per_capita_exposure = int(numpy.ceil(exposure / pop * kde_factor))
        x_exposure = [numpy.random.choice(x[(x > (p - threshold)) & (x < (p + threshold))], size=per_capita_exposure,
                                          p=kernel.pdf(x[(x > (p - threshold)) & (x < (p + threshold))]) / kernel.pdf(
                                              x[(x > (p - threshold)) & (x < (p + threshold))]).sum(), replace=True)
                      for p in x_sample]
        percentile = 50
        cross_median_exposures = numpy.hstack([
            [(p > numpy.percentile(x_sample, percentile)).sum() for p in
             numpy.array(x_exposure)[x_sample < numpy.percentile(x_sample, percentile)]],
            [(p < numpy.percentile(x_sample, percentile)).sum() for p in
             numpy.array(x_exposure)[x_sample > numpy.percentile(x_sample, percentile)]]]).mean()
        cross_median_exposures_90_10 = numpy.hstack([
            [(p > numpy.percentile(x_sample, 90)).sum() for p in
             numpy.array(x_exposure)[x_sample < numpy.percentile(x_sample, percentile)]],
            [(p < numpy.percentile(x_sample, 10)).sum() for p in
             numpy.array(x_exposure)[x_sample > numpy.percentile(x_sample, percentile)]]]).mean()

        print("done with %d" % i)
        return pop, cross_median_exposures, exposure, per_capita_exposure, threshold, numpy.percentile(x_sample,
                                                                                                       percentile), pearsonr(
            x_sample, [numpy.mean(p) for p in x_exposure]).statistic, xmax, xmin, cross_median_exposures_90_10
    except:
        print("done with %d" % i)
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan


if not Path('kde_data.pkl').exists():
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(draw_kde_one_city, i) for i in range(len(exposures))]

    pop, cross_median_exposures, exposure, per_capita_exposure, threshold, percentile, r, xmax, xmin, cross_median_exposures_90_10 = zip(
        *[f.result() for f in futures])
    kde_data = {
        'pop': pop,
        'cross_median_exposures': cross_median_exposures,
        'exposure': exposure,
        'per_capita_exposure': per_capita_exposure,
        'threshold': threshold,
        'percentile': percentile,
        'r': r,
        'xmax': xmax,
        'xmin': xmin,
        'cross_median_exposures_90_10': cross_median_exposures_90_10,
    }
    with open('kde_data.pkl', 'wb') as handle:
        pickle.dump(kde_data, handle)
else:
    with open('kde_data.pkl', 'rb') as handle:
        kde_data = pickle.load(handle)



fit_seg = OLS(numpy.log(exposures['ExposureSegregation']),
          add_constant(numpy.log(exposures['Pop.Size'].str.replace(',', '').astype(float)))).fit()
fitSES = OLS(numpy.log(exposures['MeanSES'].str.replace(',', '').astype(float)),
             add_constant(numpy.log(exposures['Pop.Size'].str.replace(',', '').astype(float)))).fit()
ynum = exposures['#Exposures'].astype(float)/exposures['Pop.Size'].str.replace(',', '').astype(float)

fit = OLS(numpy.log(ynum),add_constant(numpy.log(exposures['Pop.Size'].str.replace(',', '').astype(float)))).fit()
plt.clf()
plt.subplot(2,2,2)
plt.scatter(exposures['Pop.Size'].str.replace(',', '').astype(float),
            ynum/numpy.exp(fit.predict([1,fit.model.exog[:,1].min()])), alpha=.15,marker='.',label='# Exposures')
plt.scatter(exposures['Pop.Size'].str.replace(',', '').astype(float),
            exposures['ExposureSegregation']/numpy.exp(fit_seg.predict([1,fit_seg.model.exog[:,1].min()])), alpha=.15,marker='^',label='Segregation')
plt.plot(numpy.exp(fit_seg.model.exog[:,1]),numpy.exp(fit_seg.predict())/numpy.exp(fit_seg.predict([1,fit_seg.model.exog[:,1].min()])),color='black',linestyle='-')
plt.plot(numpy.exp(fit.model.exog[:,1]),numpy.exp(fit.predict())/numpy.exp(fit.predict([1,fit.model.exog[:,1].min()])),color='black',linestyle='-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Population')
plt.ylabel('# Exposures/Segregation\n(centered)')
plt.text(0.05, 0.64, r'$\beta_{exposures} = %.2f\pm%.2f$' % (fit.params[1], fit.bse[1]), transform=plt.gca().transAxes, rotation=17)
plt.text(0.3, 0.32, r'$\beta_{segregation} = %.2f\pm%.2f$' % (fit_seg.params[1], fit_seg.bse[1]), transform=plt.gca().transAxes, rotation=6)
plt.legend(loc='lower right')
plt.tight_layout()
#########################
plt.subplot(2,2,4)
numpy.random.seed(107)
i=309
msa = int(exposures['cbsa_code'].values[i])
income_list = income_lists[msa]
x = np.linspace(.8 * min(income_list), 1.5 * max(income_list), 1000)
income_list = np.array(income_list)
pop = np.array(fullpops[msa]) / sum(fullpops[msa])
kernel = scipy.stats.gaussian_kde(income_list, weights=pop, bw_method=wide_silverman)
threshold = kde_data['threshold'][i]
x_sample = numpy.random.choice(x, size=500, p=kernel.pdf(x) / kernel.pdf(x).sum(), replace=True)
x_exposure = [numpy.random.choice(x[(x > (p - threshold)) & (x < (p + threshold))], size=kde_data['per_capita_exposure'][i],
                                    p=kernel.pdf(x[(x > (p - threshold)) & (x < (p + threshold))]) / kernel.pdf(
                                        x[(x > (p - threshold)) & (x < (p + threshold))]).sum(), replace=True)
                for p in x_sample]
fit = OLS([numpy.mean(p) for p in x_exposure],
          add_constant(x_sample)).fit()
plt.plot(x,kernel.pdf(x),color='k',alpha=.5)
plt.xlabel('Income')
plt.ylabel('Density')
plt.xlim(15000,250000)
m = 110000-threshold
plt.plot([m,m],[.3e-5,kernel.pdf(m)[0]],color='k',linestyle='--')
plt.plot([m,m],[.0e-5,.1e-5],color='k',linestyle='--')
plt.plot([threshold+m,threshold+m],[.0e-5,kernel.pdf(threshold+m)[0]],color='k',linestyle='-')
plt.plot([2*threshold+m,2*threshold+m],[.0e-5,kernel.pdf(2*threshold+m)[0]],color='k',linestyle='--')
plt.xticks([m,50000,100000,m+threshold,150000,m+2*threshold],["\nx-t","50k","100k",'\nx',"150k",'\nx+t'])
plt.text(0.02,0.13,'Santa Fe, NM',transform=plt.gca().transAxes)
rsim = pearsonr(x_sample,[numpy.mean(p) for p in x_exposure])
r = exposures['ExposureSegregation'].values[i]
ax = plt.axes([0.815, 0.26, .15, .2])
ax.scatter(x_sample,[numpy.mean(p) for p in x_exposure],alpha=.03,color='k')
ax.set_yticks([60000,70000])
ax.set_yticklabels(['60k','70k'])
ax.set_xticks([50000,100000])
ax.set_xticklabels(['50k','100k'])
ax.set_xlabel('Simulated Inc.')
ax.set_ylabel('Simulated\nExp. Inc.')
ax.plot([numpy.nanmin(x_sample),numpy.nanmax(x_sample)],
         [fit.predict([1,numpy.nanmin(x_sample)]), fit.predict([1,numpy.nanmax(x_sample)])],
            color='black',linestyle='--')
plt.tight_layout()
plt.savefig("figures/segregation_response_small.png",dpi=400)


fit = OLS(numpy.log(kde_data['cross_median_exposures'])[~numpy.isnan(kde_data['r'])],
          add_constant(numpy.log(kde_data['pop'])[~numpy.isnan(kde_data['r'])])).fit()
plt.clf()
plt.scatter(kde_data['pop'],kde_data['cross_median_exposures'],alpha=.25,color='k')
plt.xscale('log')
plt.xlabel('MSA Population')
plt.ylabel('Per Capita Exposures Across Median')
plt.yscale('log')
plt.yticks([20,30,40,50,60,70,80,90,100,200,300,400,500,600,700],[20,"",'',50,'','','','',100,200,'','','',600,''])
plt.text(0.65,0.55,r'$\beta = %.2f\pm %.2f$' % (fit.params[1],fit.bse[1]),transform=plt.gca().transAxes,size=16, rotation=20)
plt.plot([numpy.nanmin(kde_data['pop']),numpy.nanmax(kde_data['pop'])],
         numpy.exp([fit.predict([1,numpy.log(numpy.nanmin(kde_data['pop']))]), fit.predict([1,numpy.log(numpy.nanmax(kde_data['pop']))])]),
            color='black',linestyle='--')
plt.plot([numpy.nanmin(kde_data['pop']),numpy.nanmax(kde_data['pop'])],
         numpy.exp([fit.predict([1,numpy.log(numpy.nanmin(kde_data['pop']))]), fit.predict([1,numpy.log(numpy.nanmin(kde_data['pop']))])]),
            color='black',linestyle='--',alpha=.5)
plt.tight_layout()
plt.savefig("figures/segregation_response_large.png",dpi=400)
