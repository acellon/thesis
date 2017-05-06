import numpy as np
import chb
import matplotlib.pyplot as plt
import pickle
from helpers import *


def good_init(fignum):
    chb05 = np.load('./outputs/chb05a5long.npz', encoding='latin1')
    print('chb05:', int(len(chb05.files) / 2))
    chb11 = np.load('./outputs/chb11init.npz', encoding='latin1')
    print('chb11:', int(len(chb11.files) / 2))
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    print('chb17:', int(len(chb17.files) / 2))
    chb20 = np.load('./outputs/chb20init.npz', encoding='latin1')
    print('chb20:', int(len(chb20.files) / 2))
    chb21 = np.load('./outputs/chb21init.npz', encoding='latin1')
    print('chb21:', int(len(chb21.files) / 2))
    chb22 = np.load('./outputs/chb22init.npz', encoding='latin1')
    print('chb22:', int(len(chb22.files) / 2))

    figA = plt.figure(fignum, figsize=(24, 20))
    ax1 = plt.subplot(5, 6, 1)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.plot(chb05['prob_1'])
    plt.plot(chb05['true_1'])
    plt.subplot(5, 6, 2)
    plt.plot(chb05['prob_2'])
    plt.plot(chb05['true_2'])
    plt.subplot(5, 6, 3)
    plt.plot(chb05['prob_3'])
    plt.plot(chb05['true_3'])
    plt.subplot(5, 6, 4)
    plt.plot(chb05['prob_4'])
    plt.plot(chb05['true_4'])
    plt.subplot(5, 6, 5)
    plt.plot(chb05['prob_5'])
    plt.plot(chb05['true_5'])

    ax2 = plt.subplot(5, 6, 7)
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=20, weight='bold')
    plt.plot(chb11['prob_1'])
    plt.plot(chb11['true_1'])
    plt.subplot(5, 6, 8)
    plt.plot(chb11['prob_2'])
    plt.plot(chb11['true_2'])
    plt.subplot(5, 6, 9)
    plt.plot(chb11['prob_3'])
    plt.plot(chb11['true_3'])

    ax3 = plt.subplot(5, 6, 13)
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=20, weight='bold')
    plt.plot(chb20['prob_1'])
    plt.plot(chb20['true_1'])
    plt.subplot(5, 6, 14)
    plt.plot(chb20['prob_2'])
    plt.plot(chb20['true_2'])
    plt.subplot(5, 6, 15)
    plt.plot(chb20['prob_4'])
    plt.plot(chb20['true_4'])
    plt.subplot(5, 6, 16)
    plt.plot(chb20['prob_6'])
    plt.plot(chb20['true_6'])
    plt.subplot(5, 6, 17)
    plt.plot(chb20['prob_7'])
    plt.plot(chb20['true_7'])
    plt.subplot(5, 6, 18)
    plt.plot(chb20['prob_8'])
    plt.plot(chb20['true_8'])

    ax4 = plt.subplot(5, 6, 19)
    ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, size=20, weight='bold')
    plt.plot(chb21['prob_1'])
    plt.plot(chb21['true_1'])
    plt.subplot(5, 6, 20)
    plt.plot(chb21['prob_2'])
    plt.plot(chb21['true_2'])
    plt.subplot(5, 6, 21)
    plt.plot(chb21['prob_3'])
    plt.plot(chb21['true_3'])
    plt.subplot(5, 6, 22)
    plt.plot(chb21['prob_4'])
    plt.plot(chb21['true_4'])

    ax5 = plt.subplot(5, 6, 25)
    ax5.text(-0.1, 1.05, 'E', transform=ax5.transAxes, size=20, weight='bold')
    plt.plot(chb22['prob_1'])
    plt.plot(chb22['true_1'])
    plt.subplot(5, 6, 26)
    plt.plot(chb22['prob_2'])
    plt.plot(chb22['true_2'])
    plt.subplot(5, 6, 27)
    plt.plot(chb22['prob_3'])
    plt.plot(chb22['true_3'])

    fignum += 1
    return fignum


def okay_init(fignum):
    chb01 = np.load('./outputs/chb01us.npz', encoding='latin1')
    print('chb01:', int(len(chb01.files) / 2))
    chb03 = np.load('./outputs/chb03init.npz', encoding='latin1')
    print('chb03:', int(len(chb03.files) / 2))
    chb07 = np.load('./outputs/chb07init.npz', encoding='latin1')
    print('chb07:', int(len(chb07.files) / 2))
    chb08 = np.load('./outputs/chb08init.npz', encoding='latin1')
    print('chb08:', int(len(chb08.files) / 2))
    chb10 = np.load('./outputs/chb10init.npz', encoding='latin1')
    print('chb10:', int(len(chb10.files) / 2))
    chb17 = np.load('./outputs/chb17init.npz', encoding='latin1')
    print('chb17:', int(len(chb17.files) / 2))
    chb18 = np.load('./outputs/chb18init.npz', encoding='latin1')
    print('chb18:', int(len(chb18.files) / 2))
    chb19 = np.load('./outputs/chb19init.npz', encoding='latin1')
    print('chb19:', int(len(chb19.files) / 2))

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

    fignum += 1
    return fignum


def bad_init(fignum):
    chb02 = np.load('./outputs/chb02init.npz', encoding='latin1')
    print('chb02:', int(len(chb02.files) / 2))
    chb04 = np.load('./outputs/chb04init.npz', encoding='latin1')
    print('chb04:', int(len(chb04.files) / 2))
    chb09 = np.load('./outputs/chb09us3.npz', encoding='latin1')
    print('chb09:', int(len(chb09.files) / 2))

    figA = plt.figure(fignum, figsize=(16, 12))
    ax1 = plt.subplot(3, 4, 1)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, size=20, weight='bold')
    plt.plot(chb02['prob_1'])
    plt.plot(chb02['true_1'])
    plt.subplot(3, 4, 2)
    plt.plot(chb02['prob_2'])
    plt.plot(chb02['true_2'])

    ax2 = plt.subplot(3, 4, 5)
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, size=20, weight='bold')
    plt.plot(chb04['prob_1'])
    plt.plot(chb04['true_1'])
    plt.subplot(3, 4, 6)
    plt.plot(chb04['prob_2'])
    plt.plot(chb04['true_2'])
    plt.subplot(3, 4, 7)
    plt.plot(chb04['prob_3'])
    plt.plot(chb04['true_3'])
    plt.subplot(3, 4, 8)
    plt.plot(chb04['prob_4'])
    plt.plot(chb04['true_4'])

    ax3 = plt.subplot(3, 4, 9)
    ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, size=20, weight='bold')
    plt.plot(chb09['prob_1'])
    plt.plot(chb09['true_1'])
    plt.subplot(3, 4, 10)
    plt.plot(chb09['prob_2'])
    plt.plot(chb09['true_2'])
    plt.subplot(3, 4, 11)
    plt.plot(chb09['prob_3'])
    plt.plot(chb09['true_3'])
    plt.subplot(3, 4, 12)
    plt.plot(chb09['prob_4'])
    plt.plot(chb09['true_4'])

    # save it
    fignum += 1
    return fignum


def chb05samptest(fignum):
    chb05samp = pickle.load(open('chb05samptest.p', 'rb'))
    osr5 = [test['osr'] for test in chb05samp]
    usp5 = [test['usp'] for test in chb05samp]
    mcc5 = [np.mean(test['mcc']) for test in chb05samp]
    mccstd5 = [np.std(test['mcc']) for test in chb05samp]
    #pc5 = [szr_pc(chb05, test['osr'], test['usp']) for test in chb05samp]
    pc5 = [
        7.055728704718547, 0.30496601807595364, 0.2595324600583415,
        0.18818189241869177, 0.25793116512618486, 2.1601597219729216,
        0.13654413249072742, 0.09967693555044878, 0.20655281033737857,
        0.22033983540279306, 0.996377360265537, 0.8707287925490147,
        0.42850741038381307, 0.311904003015013, 1.2209705765585614
    ]
    figA = plt.figure(fignum)
    plt.scatter(usp5, osr5, c=mcc5, s=[x * 5e3 for x in mccstd5], alpha=0.75)
    plt.ylabel('Oversampling rate')
    plt.xlabel('Probability of undersampling')
    plt.colorbar()
    #save it
    fignum += 1

    figB = plt.figure(fignum)
    ax = plt.gca()
    ax.scatter(
        pc5, mcc5, marker='o', s=[x * 2.5e3 for x in mccstd5], alpha=0.75)
    ax.set_xscale('log')
    plt.xlabel('Seizure/Nonseizure training ratio')
    plt.ylabel('Mean MCC')
    ax.set_axisbelow(True)
    ax.grid(which='both')
    # save it
    fignum += 1
    return fignum


def chb20samptest(fignum):
    chb20samp = pickle.load(open('chb20samptest.p', 'rb'))
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
        pc20, mcc20, marker='o', s=[x * 2e3 for x in mccstd20], alpha=0.75)
    ax.set_xscale('log')
    plt.xlabel('Seizure/Nonseizure training ratio')
    plt.ylabel('Mean MCC')
    ax.set_axisbelow(True)
    ax.grid(which='both')
    #save it
    fignum += 1
    return fignum


def chb21samptest(fignum):
    chb21samp = pickle.load(open('chb21samptest.p', 'rb'))
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
    #save it
    fignum += 1

    figB = plt.figure(fignum)
    ax = plt.gca()
    ax.scatter(
        pc21, mcc21, marker='o', s=[x * 2e3 for x in mccstd21], alpha=0.75)
    #ax.set_xscale('log')
    plt.xlabel('Seizure/Nonseizure training ratio')
    plt.ylabel('Mean MCC')
    ax.set_axisbelow(True)
    ax.grid(which='both')
    #save it
    fignum += 1
    return fignum
