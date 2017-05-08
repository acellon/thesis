import matplotlib as mpl
mpl.use('pgf')

import numpy as np

def figsize(scale):
    fig_width_pt = 345.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {
    "pgf.texsystem":
    "pdflatex",
    "text.usetex":
    True,
    "font.family":
    "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize":
    8,
    "axes.titlesize":
    8,
    "font.size":
    10,
    "legend.fontsize":
    8,
    "xtick.labelsize":
    8,
    "ytick.labelsize":
    8,
    "lines.linewidth":
    0.75,
    "figure.figsize":
    figsize(0.9),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import chb
import pickle
from helpers import *


def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width))
    return fig


def savefig(filename):
    plt.savefig('../report/img/{}.pgf'.format(filename))
    plt.savefig('../report/img/{}.pdf'.format(filename))


def good_init():
    chb05 = np.load('./outputs/chb05a5long.npz', encoding='latin1')
    # print('chb05:', int(len(chb05.files) / 2))
    chb11 = np.load('./outputs/chb11init.npz', encoding='latin1')
    # print('chb11:', int(len(chb11.files) / 2))
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    # print('chb17:', int(len(chb17.files) / 2))
    chb20 = np.load('./outputs/chb20init.npz', encoding='latin1')
    # print('chb20:', int(len(chb20.files) / 2))
    chb21 = np.load('./outputs/chb21init.npz', encoding='latin1')
    # print('chb21:', int(len(chb21.files) / 2))
    chb22 = np.load('./outputs/chb22init.npz', encoding='latin1')
    # print('chb22:', int(len(chb22.files) / 2))

    figA = newfig(1)
    ax1 = plt.subplot(6, 5, 1)
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=10, weight='bold')
    ax1.plot(chb05['prob_1'])
    ax1.plot(chb05['true_1'])
    ax6 = plt.subplot(6, 5, 6)
    ax6.plot(chb05['prob_2'])
    ax6.plot(chb05['true_2'])
    ax11 = plt.subplot(6, 5, 11)
    ax11.plot(chb05['prob_3'])
    ax11.plot(chb05['true_3'])
    ax16 = plt.subplot(6, 5, 16)
    ax16.plot(chb05['prob_4'])
    ax16.plot(chb05['true_4'])
    ax21 = plt.subplot(6, 5, 21)
    ax21.plot(chb05['prob_5'])
    ax21.plot(chb05['true_5'])

    ax2 = plt.subplot(6, 5, 2)
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=10, weight='bold')
    plt.plot(chb11['prob_1'])
    plt.plot(chb11['true_1'])
    plt.subplot(6, 5, 7)
    plt.plot(chb11['prob_2'])
    plt.plot(chb11['true_2'])
    plt.subplot(6, 5, 12)
    plt.plot(chb11['prob_3'])
    plt.plot(chb11['true_3'])

    ax3 = plt.subplot(6, 5, 3)
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=10, weight='bold')
    plt.plot(chb20['prob_1'])
    plt.plot(chb20['true_1'])
    plt.subplot(6, 5, 8)
    plt.plot(chb20['prob_2'])
    plt.plot(chb20['true_2'])
    plt.subplot(6, 5, 13)
    plt.plot(chb20['prob_4'])
    plt.plot(chb20['true_4'])
    plt.subplot(6, 5, 18)
    plt.plot(chb20['prob_6'])
    plt.plot(chb20['true_6'])
    plt.subplot(6, 5, 23)
    plt.plot(chb20['prob_7'])
    plt.plot(chb20['true_7'])
    plt.subplot(6, 5, 28)
    plt.plot(chb20['prob_8'])
    plt.plot(chb20['true_8'])

    ax4 = plt.subplot(6, 5, 4)
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, size=10, weight='bold')
    plt.plot(chb21['prob_1'])
    plt.plot(chb21['true_1'])
    plt.subplot(6, 5, 9)
    plt.plot(chb21['prob_2'])
    plt.plot(chb21['true_2'])
    plt.subplot(6, 5, 14)
    plt.plot(chb21['prob_3'])
    plt.plot(chb21['true_3'])
    plt.subplot(6, 5, 19)
    plt.plot(chb21['prob_4'])
    plt.plot(chb21['true_4'])

    ax5 = plt.subplot(6, 5, 5)
    ax5.text(-0.1, 1.1, 'E', transform=ax5.transAxes, size=10, weight='bold')
    plt.plot(chb22['prob_1'])
    plt.plot(chb22['true_1'])
    plt.subplot(6, 5, 10)
    plt.plot(chb22['prob_2'])
    plt.plot(chb22['true_2'])
    plt.subplot(6, 5, 15)
    plt.plot(chb22['prob_3'])
    plt.plot(chb22['true_3'])

    plt.tight_layout()
    savefig('good_init')


def good_init2():
    chb05 = np.load('./outputs/chb05a5long.npz', encoding='latin1')
    chb08 = np.load('./outputs/chb08init.npz', encoding='latin1')
    chb11 = np.load('./outputs/chb11init.npz', encoding='latin1')
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    chb20 = np.load('./outputs/chb20init.npz', encoding='latin1')
    chb21 = np.load('./outputs/chb21init.npz', encoding='latin1')
    chb22 = np.load('./outputs/chb22init.npz', encoding='latin1')

    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.05, hspace=0.4)
    figA = newfig(1)

    ax00 = plt.subplot(gs[0, 0])
    ax00.text(-0.5, 1.1, 'A', transform=ax00.transAxes, size=10, weight='bold')
    ax00.set_title('Seizure 1')
    ax00.set_ylabel('$P(seizure)$')
    ax00.plot(chb05['prob_1'])
    ax00.plot(chb05['true_1'])
    ax01 = plt.subplot(gs[0, 1])
    ax01.get_yaxis().set_visible(False)
    ax01.set_title('Seizure 2')
    ax01.plot(chb05['prob_2'])
    ax01.plot(chb05['true_2'])
    ax02 = plt.subplot(gs[0, 2])
    ax02.get_yaxis().set_visible(False)
    ax02.set_title('Seizure 3')
    ax02.plot(chb05['prob_3'])
    ax02.plot(chb05['true_3'])
    ax03 = plt.subplot(gs[0, 3])
    ax03.get_yaxis().set_visible(False)
    ax03.set_title('Seizure 4')
    ax03.plot(chb05['prob_4'])
    ax03.plot(chb05['true_4'])
    ax04 = plt.subplot(gs[0, 4])
    ax04.get_yaxis().set_visible(False)
    ax04.set_title('Seizure 5')
    ax04.plot(chb05['prob_5'])
    ax04.plot(chb05['true_5'])

    ax10 = plt.subplot(gs[1,0])
    ax10.text(-0.5, 1.1, 'B', transform=ax10.transAxes, size=10, weight='bold')
    ax10.set_ylabel('$P(seizure)$')
    ax10.plot(chb08['prob_1'])
    ax10.plot(chb08['true_1'])
    ax11 = plt.subplot(gs[1,1])
    ax11.get_yaxis().set_visible(False)
    ax11.plot(chb08['prob_2'])
    ax11.plot(chb08['true_2'])
    ax12 = plt.subplot(gs[1,2])
    ax12.get_yaxis().set_visible(False)
    ax12.plot(chb08['prob_3'])
    ax12.plot(chb08['true_3'])
    ax13 = plt.subplot(gs[1,3])
    ax13.get_yaxis().set_visible(False)
    ax13.plot(chb08['prob_4'])
    ax13.plot(chb08['true_4'])
    ax14 = plt.subplot(gs[1,4])
    ax14.get_yaxis().set_visible(False)
    ax14.plot(chb08['prob_5'])
    ax14.plot(chb08['true_5'])
    '''
    ax10 = plt.subplot(gs[1, 0])
    ax10.text(-0.5, 1.1, 'B', transform=ax10.transAxes, size=10, weight='bold')
    ax10.plot(chb11['prob_1'])
    ax10.plot(chb11['true_1'])
    ax11 = plt.subplot(gs[1, 1])
    ax11.get_yaxis().set_visible(False)
    ax11.plot(chb11['prob_2'])
    ax11.plot(chb11['true_2'])
    ax12 = plt.subplot(gs[1, 2])
    ax12.get_yaxis().set_visible(False)
    ax12.plot(chb11['prob_3'])
    ax12.plot(chb11['true_3'])

    ax20 = plt.subplot(gs[2, 0])
    ax20.text(-0.5, 1.1, 'C', transform=ax20.transAxes, size=10, weight='bold')
    ax20.plot(chb21['prob_1'])
    ax20.plot(chb21['true_1'])
    ax21 = plt.subplot(gs[2, 1])
    ax21.get_yaxis().set_visible(False)
    ax21.plot(chb21['prob_2'])
    ax21.plot(chb21['true_2'])
    ax22 = plt.subplot(gs[2, 2])
    ax22.get_yaxis().set_visible(False)
    ax22.plot(chb21['prob_3'])
    ax22.plot(chb21['true_3'])
    ax23 = plt.subplot(gs[2, 3])
    ax23.get_yaxis().set_visible(False)
    ax23.plot(chb21['prob_4'])
    ax23.plot(chb21['true_4'])
    '''
    savefig('good_init2')


def okay_init():
    chb01 = np.load('./outputs/chb01us.npz', encoding='latin1')
    # print('chb01:', int(len(chb01.files) / 2))
    chb03 = np.load('./outputs/chb03init.npz', encoding='latin1')
    # print('chb03:', int(len(chb03.files) / 2))
    chb07 = np.load('./outputs/chb07init.npz', encoding='latin1')
    # print('chb07:', int(len(chb07.files) / 2))
    chb08 = np.load('./outputs/chb08init.npz', encoding='latin1')
    # print('chb08:', int(len(chb08.files) / 2))
    chb10 = np.load('./outputs/chb10init.npz', encoding='latin1')
    # print('chb10:', int(len(chb10.files) / 2))
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    # print('chb17:', int(len(chb17.files) / 2))
    chb18 = np.load('./outputs/chb18init.npz', encoding='latin1')
    # print('chb18:', int(len(chb18.files) / 2))
    chb19 = np.load('./outputs/chb19init.npz', encoding='latin1')
    # print('chb19:', int(len(chb19.files) / 2))

    figA = plt.figure(fignum, figsize=(24, 24))
    ax1 = plt.subplot(8, 7, 1)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.plot(chb01['prob_1'])
    plt.plot(chb01['true_1'])
    plt.subplot(8, 7, 2)
    plt.plot(chb01['prob_2'])
    plt.plot(chb01['true_2'])
    plt.subplot(8, 7, 3)
    plt.plot(chb01['prob_3'])
    plt.plot(chb01['true_3'])
    plt.subplot(8, 7, 4)
    plt.plot(chb01['prob_4'])
    plt.plot(chb01['true_4'])
    plt.subplot(8, 7, 5)
    plt.plot(chb01['prob_5'])
    plt.plot(chb01['true_5'])
    plt.subplot(8, 7, 6)
    plt.plot(chb01['prob_6'])
    plt.plot(chb01['true_6'])
    plt.subplot(8, 7, 7)
    plt.plot(chb01['prob_7'])
    plt.plot(chb01['true_7'])

    ax2 = plt.subplot(8, 7, 8)
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=20, weight='bold')
    plt.plot(chb03['prob_1'])
    plt.plot(chb03['true_1'])
    plt.subplot(8, 7, 9)
    plt.plot(chb03['prob_2'])
    plt.plot(chb03['true_2'])
    plt.subplot(8, 7, 10)
    plt.plot(chb03['prob_3'])
    plt.plot(chb03['true_3'])
    plt.subplot(8, 7, 11)
    plt.plot(chb03['prob_4'])
    plt.plot(chb03['true_4'])
    plt.subplot(8, 7, 12)
    plt.plot(chb03['prob_5'])
    plt.plot(chb03['true_5'])
    plt.subplot(8, 7, 13)
    plt.plot(chb03['prob_6'])
    plt.plot(chb03['true_6'])
    plt.subplot(8, 7, 14)
    plt.plot(chb03['prob_7'])
    plt.plot(chb03['true_7'])

    ax3 = plt.subplot(8, 7, 15)
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=20, weight='bold')
    plt.plot(chb07['prob_1'])
    plt.plot(chb07['true_1'])
    plt.subplot(8, 7, 16)
    plt.plot(chb07['prob_2'])
    plt.plot(chb07['true_2'])
    plt.subplot(8, 7, 17)
    plt.plot(chb07['prob_3'])
    plt.plot(chb07['true_3'])

    ax4 = plt.subplot(8, 7, 22)
    ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, size=20, weight='bold')
    plt.plot(chb08['prob_1'])
    plt.plot(chb08['true_1'])
    plt.subplot(8, 7, 23)
    plt.plot(chb08['prob_2'])
    plt.plot(chb08['true_2'])
    plt.subplot(8, 7, 24)
    plt.plot(chb08['prob_3'])
    plt.plot(chb08['true_3'])
    plt.subplot(8, 7, 25)
    plt.plot(chb08['prob_4'])
    plt.plot(chb08['true_4'])
    plt.subplot(8, 7, 26)
    plt.plot(chb08['prob_5'])
    plt.plot(chb08['true_5'])

    ax5 = plt.subplot(8, 7, 29)
    ax5.text(-0.1, 1.05, 'E', transform=ax5.transAxes, size=20, weight='bold')
    plt.plot(chb10['prob_1'])
    plt.plot(chb10['true_1'])
    plt.subplot(8, 7, 30)
    plt.plot(chb10['prob_2'])
    plt.plot(chb10['true_2'])
    plt.subplot(8, 7, 31)
    plt.plot(chb10['prob_3'])
    plt.plot(chb10['true_3'])
    plt.subplot(8, 7, 32)
    plt.plot(chb10['prob_4'])
    plt.plot(chb10['true_4'])
    plt.subplot(8, 7, 33)
    plt.plot(chb10['prob_5'])
    plt.plot(chb10['true_5'])
    plt.subplot(8, 7, 34)
    plt.plot(chb10['prob_6'])
    plt.plot(chb10['true_6'])
    plt.subplot(8, 7, 35)
    plt.plot(chb10['prob_7'])
    plt.plot(chb10['true_7'])

    ax6 = plt.subplot(8, 7, 36)
    ax6.text(-0.1, 1.05, 'F', transform=ax6.transAxes, size=20, weight='bold')
    plt.plot(chb17['prob_1'])
    plt.plot(chb17['true_1'])
    plt.subplot(8, 7, 37)
    plt.plot(chb17['prob_2'])
    plt.plot(chb17['true_2'])
    plt.subplot(8, 7, 38)
    plt.plot(chb17['prob_3'])
    plt.plot(chb17['true_3'])

    ax7 = plt.subplot(8, 7, 43)
    ax7.text(-0.1, 1.05, 'G', transform=ax7.transAxes, size=20, weight='bold')
    plt.plot(chb18['prob_1'])
    plt.plot(chb18['true_1'])
    plt.subplot(8, 7, 44)
    plt.plot(chb18['prob_2'])
    plt.plot(chb18['true_2'])
    plt.subplot(8, 7, 45)
    plt.plot(chb18['prob_3'])
    plt.plot(chb18['true_3'])
    plt.subplot(8, 7, 46)
    plt.plot(chb18['prob_4'])
    plt.plot(chb18['true_4'])
    plt.subplot(8, 7, 47)
    plt.plot(chb18['prob_5'])
    plt.plot(chb18['true_5'])
    plt.subplot(8, 7, 48)
    plt.plot(chb18['prob_6'])
    plt.plot(chb18['true_6'])

    ax8 = plt.subplot(8, 7, 50)
    ax8.text(-0.1, 1.05, 'H', transform=ax8.transAxes, size=20, weight='bold')
    plt.plot(chb19['prob_1'])
    plt.plot(chb19['true_1'])
    plt.subplot(8, 7, 51)
    plt.plot(chb19['prob_2'])
    plt.plot(chb19['true_2'])
    plt.subplot(8, 7, 52)
    plt.plot(chb19['prob_3'])
    plt.plot(chb19['true_3'])

    savefig('okay_init')


def bad_init():
    chb02 = np.load('./outputs/chb02init.npz', encoding='latin1')
    # print('chb02:', int(len(chb02.files) / 2))
    chb04 = np.load('./outputs/chb04init.npz', encoding='latin1')
    # print('chb04:', int(len(chb04.files) / 2))
    chb09 = np.load('./outputs/chb09us3.npz', encoding='latin1')
    # print('chb09:', int(len(chb09.files) / 2))

    gs = gridspec.GridSpec(3, 4)
    gs.update(hspace=0.4)
    figA = newfig(1)

    ax00 = plt.subplot(gs[0,0])
    ax00.text(-0.5, 1.1, 'A', transform=ax00.transAxes, size=10, weight='bold')
    ax00.set_title('Seizure 1')
    ax00.set_ylabel('$P(seizure)$')
    ax00.plot(chb02['prob_1'])
    ax00.plot(chb02['true_1'])
    ax01 = plt.subplot(gs[0,1])
    ax01.get_yaxis().set_visible(False)
    ax01.set_title('Seizure 2')
    ax01.plot(chb02['prob_2'])
    ax01.plot(chb02['true_2'])

    ax10 = plt.subplot(gs[1,0])
    ax10.text(-0.5, 1.1, 'B', transform=ax10.transAxes, size=10, weight='bold')
    ax10.set_ylabel('$P(seizure)$')
    ax10.plot(chb04['prob_1'])
    ax10.plot(chb04['true_1'])
    ax11 = plt.subplot(gs[1,1])
    ax11.get_yaxis().set_visible(False)
    ax11.plot(chb04['prob_2'])
    ax11.plot(chb04['true_2'])
    ax12 = plt.subplot(gs[1,2])
    ax12.get_yaxis().set_visible(False)
    ax12.set_title('Seizure 3')
    ax12.plot(chb04['prob_3'])
    ax12.plot(chb04['true_3'])
    ax13 = plt.subplot(gs[1,3])
    ax13.get_yaxis().set_visible(False)
    ax13.set_title('Seizure 4')
    ax13.plot(chb04['prob_4'])
    ax13.plot(chb04['true_4'])

    ax20 = plt.subplot(gs[2,0])
    ax20.text(-0.5, 1.1, 'C', transform=ax20.transAxes, size=10, weight='bold')
    ax20.set_ylabel('$P(seizure)$')
    ax20.plot(chb09['prob_1'])
    ax20.plot(chb09['true_1'])
    ax21 = plt.subplot(gs[2,1])
    ax21.get_yaxis().set_visible(False)
    ax21.plot(chb09['prob_2'])
    ax21.plot(chb09['true_2'])
    ax22 = plt.subplot(gs[2,2])
    ax22.get_yaxis().set_visible(False)
    ax22.plot(chb09['prob_3'])
    ax22.plot(chb09['true_3'])
    ax23 = plt.subplot(gs[2,3])
    ax23.get_yaxis().set_visible(False)
    ax23.plot(chb09['prob_4'])
    ax23.plot(chb09['true_4'])

    savefig('bad_init')


def chb05samptest():
    chb05samp = pickle.load(open('./tests/chb05samptest.p', 'rb'))
    osr5 = [test['osr'] for test in chb05samp]
    usp5 = [test['usp'] for test in chb05samp]
    mcc5 = [np.mean(test['mcc']) for test in chb05samp]
    mccstd5 = [np.std(test['mcc']) for test in chb05samp]
    pc5 = [
        7.055728704718547, 0.30496601807595364, 0.2595324600583415,
        0.18818189241869177, 0.25793116512618486, 2.1601597219729216,
        0.13654413249072742, 0.09967693555044878, 0.20655281033737857,
        0.22033983540279306, 0.996377360265537, 0.8707287925490147,
        0.42850741038381307, 0.311904003015013, 1.2209705765585614
    ]

    figA = newfig(1)
    gs = gridspec.GridSpec(2,1)
    ax0 = plt.subplot(gs[0])
    ax0.text(-0.05, 1.05, 'A', transform=ax0.transAxes, size=10, weight='bold')
    scat = ax0.scatter(usp5, osr5, c=mcc5, s=[x * 1e3 for x in mccstd5], alpha=0.75)
    ax0.set_ylabel('$OSR$')
    ax0.set_ylim(2,7)
    ax0.set_xlabel('$USP$')
    ax0.set_axisbelow(True)
    ax0.grid(which='both')


    ax1 = plt.subplot(gs[1])
    ax1.text(-0.05, 1.05, 'B', transform=ax1.transAxes, size=10, weight='bold')
    ax1.scatter(
        pc5,
        mcc5,
        c=mcc5,
        marker='o',
        s=[x * 1e3 for x in mccstd5],
        alpha=0.75)
    ax1.set_xscale('log')
    ax1.set_xlabel('$\\rho$')
    ax1.set_ylabel('$\overline{MCC}$')
    ax1.set_axisbelow(True)
    ax1.grid(which='both')

    gs.tight_layout(figA)
    gs.update(right=0.8)
    cbar_ax = figA.add_axes([0.85, 0.15, 0.05, 0.8])
    figA.colorbar(scat, cax=cbar_ax)

    savefig('chb05samp')

def chb20samptest():
    chb20samp = pickle.load(open('./tests/chb20samptest.p', 'rb'))
    osr20 = [test['osr'] for test in chb20samp]
    usp20 = [test['usp'] for test in chb20samp]
    mcc20 = [np.mean(test['mcc']) for test in chb20samp]
    mccstd20 = [np.std(test['mcc']) for test in chb20samp]
    #pc20 = [szr_pc(chb20, test['osr'], test['usp']) for test in chb20samp]
    pc20 = [
        2.174020692439478, 0.17565959734915595, 0.09151069199705428,
        0.05767492845207528, 0.13525186322474497, 0.07074828289725237,
        0.0736236518997994, 0.12687610462590437, 0.0702298930362629,
        0.05756723788295078
    ]
    figA = plt.figure(fignum)
    plt.scatter(
        usp20, osr20, c=mcc20, s=[x * 2e3 for x in mccstd20], alpha=0.75)
    plt.ylabel('Oversampling rate')
    plt.xlabel('Probability of undersampling')
    plt.colorbar()
    #save it
    fignum += 1

    figB = plt.figure(fignum)
    ax = plt.gca()
    ax.scatter(
        pc20,
        mcc20,
        c=mcc20,
        marker='o',
        s=[x * 2e3 for x in mccstd20],
        alpha=0.75)
    ax.set_xscale('log')
    plt.xlabel('Seizure/Nonseizure training ratio')
    plt.ylabel('Mean MCC')
    ax.set_axisbelow(True)
    ax.grid(which='both')
    #save it
    savefig('chb20samptest')


def chb21samptest():
    chb21samp = pickle.load(open('./tests/chb21samptest.p', 'rb'))
    osr21 = [test['osr'] for test in chb21samp]
    usp21 = [test['usp'] for test in chb21samp]
    mcc21 = [np.mean(test['mcc']) for test in chb21samp]
    mccstd21 = [np.std(test['mcc']) for test in chb21samp]
    #pc21 = [szr_pc(chb21, test['osr'], test['usp']) for test in chb21samp]
    pc21 = [
        0.1805585265981609, 0.184341435658232, 0.21393038265248535,
        0.3369073634027449, 0.08478047664585985, 0.11327235557235339,
        0.18657989413336035, 0.12525102336635335, 0.11673706832160421,
        0.12295543745912449
    ]
    figA = plt.figure(fignum)
    plt.scatter(
        usp21, osr21, c=mcc21, s=[x * 2e3 for x in mccstd21], alpha=0.75)
    plt.ylabel('Oversampling rate')
    plt.xlabel('Probability of undersampling')
    plt.colorbar()

    savefig('chb21samptest1')

    figB = newfig()
    ax = plt.gca()
    ax.scatter(
        pc21,
        mcc21,
        c=mcc21,
        marker='o',
        s=[x * 2e3 for x in mccstd21],
        alpha=0.75)
    #ax.set_xscale('log')
    plt.xlabel('Seizure/Nonseizure training ratio')
    plt.ylabel('Mean MCC')
    ax.set_axisbelow(True)
    ax.grid(which='both')

    savefig('chb21samptest2')

def samp20():
    chb20samp = pickle.load(open('./tests/chb20samptest.p', 'rb'))
    osr20 = [test['osr'] for test in chb20samp]
    usp20 = [test['usp'] for test in chb20samp]
    mcc20 = [np.mean(test['mcc']) for test in chb20samp]
    mccstd20 = [np.std(test['mcc']) for test in chb20samp]
    pc20 = [
        2.174020692439478, 0.17565959734915595, 0.09151069199705428,
        0.05767492845207528, 0.13525186322474497, 0.07074828289725237,
        0.0736236518997994, 0.12687610462590437, 0.0702298930362629,
        0.05756723788295078
    ]
    chb21samp = pickle.load(open('./tests/chb21samptest.p', 'rb'))
    osr21 = [test['osr'] for test in chb21samp]
    usp21 = [test['usp'] for test in chb21samp]
    mcc21 = [np.mean(test['mcc']) for test in chb21samp]
    mccstd21 = [np.std(test['mcc']) for test in chb21samp]
    pc21 = [
        0.1805585265981609, 0.184341435658232, 0.21393038265248535,
        0.3369073634027449, 0.08478047664585985, 0.11327235557235339,
        0.18657989413336035, 0.12525102336635335, 0.11673706832160421,
        0.12295543745912449
    ]

    figA = newfig(1)
    gs = gridspec.GridSpec(2,2)

    ax00 = plt.subplot(gs[0, 0])
    ax00.text(-0.05, 1.05, 'A', transform=ax00.transAxes, size=10, weight='bold')
    scat = ax00.scatter(
        usp20, osr20, c=mcc20, s=[x * .25e3 for x in mccstd20], alpha=0.75)
    ax00.set_ylabel('$OSR$')
    ax00.set_ylim(2,7)
    ax00.set_xlabel('$USP$')
    ax00.set_axisbelow(True)
    ax00.grid(which='both')

    ax01 = plt.subplot(gs[0,1])
    ax01.scatter(
        pc20,
        mcc20,
        c=mcc20,
        marker='o',
        s=[x * .25e3 for x in mccstd20],
        alpha=0.75)
    ax01.set_xscale('log')
    ax01.set_xlabel('$\\rho$')
    ax01.set_ylabel('$\overline{MCC}$')
    ax01.set_axisbelow(True)
    ax01.grid(which='both')
    plt.colorbar(scat)

    ax10 = plt.subplot(gs[1,0])
    ax10.text(-0.05, 1.05, 'B', transform=ax10.transAxes, size=10, weight='bold')
    scat2 = ax10.scatter(
        usp21, osr21, c=mcc21, s=[x * .25e3 for x in mccstd21], alpha=0.75)
    ax10.set_ylabel('$OSR$')
    ax10.set_ylim(2,7)
    ax10.set_xlabel('$USP$')
    ax10.set_axisbelow(True)
    ax10.grid(which='both')

    ax11 = plt.subplot(gs[1,1])
    ax11.scatter(
        pc21,
        mcc21,
        c=mcc21,
        marker='o',
        s=[x * .25e3 for x in mccstd21],
        alpha=0.75)
    #ax11.set_xscale('log')
    ax11.set_xlabel('$\\rho$')
    ax11.set_ylabel('$\overline{MCC}$')
    ax11.set_axisbelow(True)
    ax11.grid(which='both')
    plt.colorbar(scat2)

    gs.tight_layout(figA)
    #gs.update(right=0.8)
    #cbar_ax = figA.add_axes([0.85, 0.15, 0.05, 0.8])
    #figA.colorbar(scat, cax=cbar_ax)

    savefig('samp20')

def chb05fcltest():
    chb05fcl = pickle.load(open('./tests/chb05fcltest.p', 'rb'))
    fcl = [test['fcl'] for test in chb05fcl]
    fclMCC = [np.mean(test['mcc']) for test in chb05fcl]
    fclMCCSTD = [np.std(test['mcc']) for test in chb05fcl]
    figA = newfig(0.75)
    ax = plt.gca()
    plt.scatter(
        fcl, fclMCC, c=fclMCC, s=[x * 3e3 for x in fclMCCSTD], alpha=0.75)
    plt.colorbar()
    ax.set_xlabel('$w(FCL)$ (# nodes)')
    ax.set_ylabel('$\overline{MCC}$')
    ax.set_axisbelow(True)
    ax.grid(which='both')
    plt.tight_layout()
    #save it
    savefig('chb05fcl')


def init_summary():
    chb01 = np.load('./outputs/chb01us.npz', encoding='latin1')
    chb02 = np.load('./outputs/chb02init.npz', encoding='latin1')
    chb03 = np.load('./outputs/chb03init.npz', encoding='latin1')
    chb04 = np.load('./outputs/chb04init.npz', encoding='latin1')
    chb05 = np.load('./outputs/chb05a5long.npz', encoding='latin1')
    chb07 = np.load('./outputs/chb07init.npz', encoding='latin1')
    chb08 = np.load('./outputs/chb08init.npz', encoding='latin1')
    chb09 = np.load('./outputs/chb09us3.npz', encoding='latin1')
    chb10 = np.load('./outputs/chb10init.npz', encoding='latin1')
    chb11 = np.load('./outputs/chb11init.npz', encoding='latin1')
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    chb18 = np.load('./outputs/chb18init.npz', encoding='latin1')
    chb19 = np.load('./outputs/chb19init.npz', encoding='latin1')
    chb20 = np.load('./outputs/chb20init.npz', encoding='latin1')
    chb21 = np.load('./outputs/chb21init.npz', encoding='latin1')
    chb22 = np.load('./outputs/chb22init.npz', encoding='latin1')

    inits = [chb01, chb02, chb03, chb04, chb05, chb07, chb08, chb09,
             chb10, chb11, chb17, chb18, chb19, chb20, chb21, chb22]
    inums = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22]

    mccmeans, mccstds = [0]*len(inums), [0]*len(inums)
    for i, init in enumerate(inits):
        _, preds, trues = npzparse(init)
        _, mcclist = supermetrics(preds, trues)
        mccmeans[i] = np.mean(mcclist)
        mccstds[i] = np.std(mcclist)

    fig = newfig(0.75)
    ax = plt.subplot(111)
    plt.scatter(
        inums, mccmeans, c=mccmeans, s=[x * 250 for x in mccstds], alpha=0.9)
    #plt.colorbar()
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(which='both')
    plt.xlabel('Case')
    plt.ylabel('$\overline{MCC}$')
    ax.set_ylim(ymax=1)
    #ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    plt.tight_layout()
    savefig('init_summary')

#good_init2()
#init_summary()
#bad_init()
#chb05samptest()
#samp20()
#chb05fcltest()
