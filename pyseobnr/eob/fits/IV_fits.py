"""
Contains the fits for the so-called waveform input values.
"""

import numpy as np


class InputValueFits:
    """
    Fits for the so-called waveform input values.

    These are the amplitude and its derivatives as well as frequency and its derivative.
    These are used in several places in the model, including NQC construction,
    merger-ringdown attachment and special calibration coefficients in some odd-m modes.
    This class wraps all necessary fits as methods. Each method returns a dict with keys
    being the desired mode.

    See also Appendix A of the [SEOBNRv5HM-notes]_ .

    Args:
        m1 (float): mass of the primary
        m2 (float): mass of the secondary
        chi1 (list): dimensionless spin components of the primary
        chi2 (list): dimensionless spin components of the secondary

    """

    def __init__(self, m1, m2, chi1, chi2):
        self.m1 = m1
        self.m2 = m2
        self.chi1 = chi1
        self.chi2 = chi2

        # Variables used in the fits
        self.q = m1 / m2
        self.nu = self.q / (1 + self.q) ** 2
        self.dm = (self.q - 1) / (self.q + 1)
        self.chiAS = (chi1[2] - chi2[2]) / 2
        self.chiS = (chi1[2] + chi2[2]) / 2
        self.chi = self.chiS + self.chiAS * self.dm / (1 - 2 * self.nu)
        self.chi33 = self.chiS * self.dm + self.chiAS
        self.chi21A = self.chiS * self.dm / (1 - 1.3 * self.nu) + self.chiAS
        self.chi44A = (1 - 5 * self.nu) * self.chiS + self.chiAS * self.dm
        self.chi21D = self.chiS * self.dm / (1 - 2 * self.nu) + self.chiAS
        self.chi44D = (1 - 7 * self.nu) * self.chiS + self.chiAS * self.dm
        # Possible additional variables
        self.nuprime = 0.25 - self.nu
        self.chi33abs = np.abs(self.chi33)
        self.chi21Aabs = np.abs(self.chi21A)
        self.chi44Aabs = np.abs(self.chi44A)

    def hsign(self):
        """
        Fits for the unsigned waveform amplitude at the attachment point.
        See Appendix A1 of the [SEOBNRv5HM-notes]_ .
        This enters in the special calibration coefficients for the (2,1), (5,5) and
        (4,3) modes and is needed to ensure the correct behaviour for cases with a
        minimum in the amplitude close to the attachment point.

        Returns:
            dict: dictionary of hsign values with keys being the desired mode
        """
        h_modes = {}
        h_modes[2, 2] = (
            71.97969776036882194603 * self.nu**4
            - 13.35761402231352157344 * self.nu**3 * self.chi
            - 46.87585958426210908101 * self.nu**3
            + 0.61988944517825661507 * self.nu**2 * self.chi**2
            + 7.19426416189229733789 * self.nu**2 * self.chi
            + 12.44040490932310127903 * self.nu**2
            + 0.43014673069078152023 * self.nu * self.chi**3
            - 1.74313546783413597652 * self.nu * self.chi
            - 0.86828935763242798274 * self.nu
            - 0.08493901280736430859 * self.chi**3
            - 0.02082621429567295401 * self.chi**2
            + 0.18693991146784910695 * self.chi
            + 1.46709663479911811557
        )

        h33_opt = (
            0.099543396526939,
            -0.46670888251595877,
            1.0577311276498613,
            0.5588076974614219,
            0.02783297472862914,
            1.9626700856890176,
            0.03625826927683684,
            -0.08837148267661124,
        )
        a0 = h33_opt[0]
        a1 = h33_opt[1]
        a2 = h33_opt[2]
        a3 = h33_opt[3]
        a4 = h33_opt[4]
        a5 = h33_opt[5]
        a6 = h33_opt[6]
        a7 = h33_opt[7]
        h_modes[3, 3] = (
            a0 + a1 * self.nu + a2 * self.nu**2
        ) * self.chi33 + self.dm * (
            a3
            + a4 * self.nu
            + a5 * self.nu**2
            + a6 * self.chi33**2
            + a7 * self.nu * self.chi33**2
        )
        h21_opt = (
            -0.43142624218620174,
            0.18934641483478895,
            -1.0679206708714077,
            -0.012705592101243156,
            0.049896513254659894,
            -0.08635569747750282,
            0.28559662955708426,
            -0.16866785018894728,
            0.03317471960170524,
        )
        a0 = h21_opt[0]
        a1 = h21_opt[1]
        a2 = h21_opt[2]
        a3 = h21_opt[3]
        a4 = h21_opt[4]
        a5 = h21_opt[5]
        a6 = h21_opt[6]
        a7 = h21_opt[7]
        a8 = h21_opt[8]
        h_modes[2, 1] = -(
            self.dm
            * (
                a0
                + a1 * self.nu
                + a2 * self.nu**2
                + a3 * self.chi21A
                + a4 * self.chi21A**2
                + a5 * self.nu * self.chi21A**2
            )
            + self.chi21A * (a6 + a7 * self.nu)
            + self.dm * a8 * self.chi21A**3
        )
        h_modes[4, 4] = (
            6.23941816830717765896 * self.nu**3
            - 1.94747289640671739086 * self.nu**2
            - 0.18016517494646497322 * self.nu * self.chi44A
            - 0.61530743422322586866 * self.nu
            + 0.0314831492638571811 * self.chi44A**2
            + 0.06393079490813786958 * self.chi44A
            + 0.26253324277501910444
        )

        h55_opt = (
            0.12546778737729275,
            -0.46214165382986366,
            1.0938116770996684,
            0.06275728238909935,
            -0.7627762053397953,
            3.965851967106404,
            -7.402839191483687,
        )
        a0 = h55_opt[0]
        a1 = h55_opt[1]
        a2 = h55_opt[2]
        a3 = h55_opt[3]
        a4 = h55_opt[4]
        a5 = h55_opt[5]
        a6 = h55_opt[6]
        h_modes[5, 5] = (
            a0 * self.dm
            + a1 * self.dm * self.nu
            + a2 * self.dm * self.nu**2
            + a3 * self.chi33
            + a4 * self.chi33 * self.nu
            + a5 * self.chi33 * self.nu**2
            + a6 * self.chi33 * self.nu**3
        )
        h_modes[3, 2] = (
            8.91777113271361798752 * self.nu**3
            - 2.19450590970282899406 * self.nu**2
            + 0.30780289195595494922 * self.nu * self.chi
            - 0.38791140278558522425 * self.nu
            + 0.02259757283590252061 * self.chi**2
            - 0.02077104373158236705 * self.chi
            + 0.15544608707074264453
        )

        h43_opt = (
            -0.02008083115201138,
            0.4365758586949771,
            -1.7380786308289482,
            0.07442005213498969,
            -0.27336437808312125,
            0.80961534629319,
            0.021931530607089078,
            -0.07155397153630519,
        )
        a0 = h43_opt[0]
        a1 = h43_opt[1]
        a2 = h43_opt[2]
        a3 = h43_opt[3]
        a4 = h43_opt[4]
        a5 = h43_opt[5]
        a6 = h43_opt[6]
        a7 = h43_opt[7]
        h_modes[4, 3] = (
            a0 + a1 * self.nu + a2 * self.nu**2
        ) * self.chi33 + self.dm * (
            a3
            + a4 * self.nu
            + a5 * self.nu**2
            + a6 * self.chi33**2
            + a7 * self.nu * self.chi33**2
        )
        return h_modes

    def habs(self):
        """
        Fits for the waveform amplitude at the attachment point.
        See Appendix A1 of the [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of habs values with keys being the desired mode
        """
        h_modes = {}
        h_modes[2, 2] = np.abs(
            71.97969776036882194603 * self.nu**4
            - 13.35761402231352157344 * self.nu**3 * self.chi
            - 46.87585958426210908101 * self.nu**3
            + 0.61988944517825661507 * self.nu**2 * self.chi**2
            + 7.19426416189229733789 * self.nu**2 * self.chi
            + 12.44040490932310127903 * self.nu**2
            + 0.43014673069078152023 * self.nu * self.chi**3
            - 1.74313546783413597652 * self.nu * self.chi
            - 0.86828935763242798274 * self.nu
            - 0.08493901280736430859 * self.chi**3
            - 0.02082621429567295401 * self.chi**2
            + 0.18693991146784910695 * self.chi
            + 1.46709663479911811557
        )

        h33_opt = (
            0.099543396526939,
            -0.46670888251595877,
            1.0577311276498613,
            0.5588076974614219,
            0.02783297472862914,
            1.9626700856890176,
            0.03625826927683684,
            -0.08837148267661124,
        )
        a0 = h33_opt[0]
        a1 = h33_opt[1]
        a2 = h33_opt[2]
        a3 = h33_opt[3]
        a4 = h33_opt[4]
        a5 = h33_opt[5]
        a6 = h33_opt[6]
        a7 = h33_opt[7]
        h_modes[3, 3] = np.abs(
            (a0 + a1 * self.nu + a2 * self.nu**2) * self.chi33
            + self.dm
            * (
                a3
                + a4 * self.nu
                + a5 * self.nu**2
                + a6 * self.chi33**2
                + a7 * self.nu * self.chi33**2
            )
        )
        h21_opt = (
            -0.43142624218620174,
            0.18934641483478895,
            -1.0679206708714077,
            -0.012705592101243156,
            0.049896513254659894,
            -0.08635569747750282,
            0.28559662955708426,
            -0.16866785018894728,
            0.03317471960170524,
        )
        a0 = h21_opt[0]
        a1 = h21_opt[1]
        a2 = h21_opt[2]
        a3 = h21_opt[3]
        a4 = h21_opt[4]
        a5 = h21_opt[5]
        a6 = h21_opt[6]
        a7 = h21_opt[7]
        a8 = h21_opt[8]
        h_modes[2, 1] = np.abs(
            -(
                self.dm
                * (
                    a0
                    + a1 * self.nu
                    + a2 * self.nu**2
                    + a3 * self.chi21A
                    + a4 * self.chi21A**2
                    + a5 * self.nu * self.chi21A**2
                )
                + self.chi21A * (a6 + a7 * self.nu)
                + self.dm * a8 * self.chi21A**3
            )
        )
        h_modes[4, 4] = np.abs(
            6.23941816830717765896 * self.nu**3
            - 1.94747289640671739086 * self.nu**2
            - 0.18016517494646497322 * self.nu * self.chi44A
            - 0.61530743422322586866 * self.nu
            + 0.0314831492638571811 * self.chi44A**2
            + 0.06393079490813786958 * self.chi44A
            + 0.26253324277501910444
        )

        h55_opt = (
            0.12546778737729275,
            -0.46214165382986366,
            1.0938116770996684,
            0.06275728238909935,
            -0.7627762053397953,
            3.965851967106404,
            -7.402839191483687,
        )
        a0 = h55_opt[0]
        a1 = h55_opt[1]
        a2 = h55_opt[2]
        a3 = h55_opt[3]
        a4 = h55_opt[4]
        a5 = h55_opt[5]
        a6 = h55_opt[6]
        h_modes[5, 5] = np.abs(
            a0 * self.dm
            + a1 * self.dm * self.nu
            + a2 * self.dm * self.nu**2
            + a3 * self.chi33
            + a4 * self.chi33 * self.nu
            + a5 * self.chi33 * self.nu**2
            + a6 * self.chi33 * self.nu**3
        )
        h_modes[3, 2] = np.abs(
            8.91777113271361798752 * self.nu**3
            - 2.19450590970282899406 * self.nu**2
            + 0.30780289195595494922 * self.nu * self.chi
            - 0.38791140278558522425 * self.nu
            + 0.02259757283590252061 * self.chi**2
            - 0.02077104373158236705 * self.chi
            + 0.15544608707074264453
        )

        h43_opt = (
            -0.02008083115201138,
            0.4365758586949771,
            -1.7380786308289482,
            0.07442005213498969,
            -0.27336437808312125,
            0.80961534629319,
            0.021931530607089078,
            -0.07155397153630519,
        )
        a0 = h43_opt[0]
        a1 = h43_opt[1]
        a2 = h43_opt[2]
        a3 = h43_opt[3]
        a4 = h43_opt[4]
        a5 = h43_opt[5]
        a6 = h43_opt[6]
        a7 = h43_opt[7]
        h_modes[4, 3] = np.abs(
            (a0 + a1 * self.nu + a2 * self.nu**2) * self.chi33
            + self.dm
            * (
                a3
                + a4 * self.nu
                + a5 * self.nu**2
                + a6 * self.chi33**2
                + a7 * self.nu * self.chi33**2
            )
        )
        return h_modes

    def hdot(self):
        """
        Fits for the waveform amplitude first derivative at the attachment point.
        See Appendix A2 of the [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of hdot values with keys being the desired mode
        """
        hdot_modes = {}
        hdot_modes[2, 2] = 0.0
        h33dot_opt = (
            -0.002094284130247092,
            0.004941495533145558,
            0.0017812020483270674,
            4.6379056832649805,
            85.17330624004674,
            -2.9868894742025103,
            39.24753775747886,
        )
        a0 = h33dot_opt[0]
        a1 = h33dot_opt[1]
        a2 = h33dot_opt[2]
        a3 = h33dot_opt[3]
        a4 = h33dot_opt[4]
        a5 = h33dot_opt[5]
        a6 = h33dot_opt[6]
        hdot_modes[3, 3] = self.dm * (
            a0 + a1 * self.nu
        ) * self.chi33**2 + a2 * np.sqrt(
            np.abs(
                self.dm**2 * (a3 + a4 * self.nu)
                + self.dm * (a5 + a6 * self.nu) * self.chi33
                + self.chi33**2
            )
        )
        h21dot_opt = (
            0.006742551578105149,
            -0.029700111038235357,
            0.008255672129873798,
            0.8154821332503585,
            1.2355885773365995,
            5.471011368680218,
            -0.00806371942464999,
            0.023534029208221174,
        )
        a0 = h21dot_opt[0]
        a1 = h21dot_opt[1]
        a2 = h21dot_opt[2]
        a3 = h21dot_opt[3]
        a4 = h21dot_opt[4]
        a5 = h21dot_opt[5]
        a6 = h21dot_opt[6]
        a7 = h21dot_opt[7]
        hdot_modes[2, 1] = (
            self.dm * (a0 + a1 * self.nu)
            + a2 * abs(-self.dm * (a3 + a4 * self.nu + a5 * self.nu**2) + self.chi21D)
            + self.dm * (a6 + a7 * self.nu) * self.chi21D
        )
        hdot_modes[4, 4] = (
            1.1346791592322915676 * self.nu**3
            - 0.03430787888412303172 * self.nu**2 * self.chi44D
            - 0.41705646774353932749 * self.nu**2
            + 0.00638741869700277728 * self.nu * self.chi44D**2
            + 0.01437277008123238192 * self.nu * self.chi44D
            + 0.02400412959637425805 * self.nu
            - 0.00125090225166799842 * self.chi44D**3
            - 0.00122253007368038871 * self.chi44D**2
            - 0.00068144676521733236 * self.chi44D
            + 0.00349811401956392001
        )

        h55dot_opt = (
            0.0025632997344277133,
            -0.01089093602552064,
            -0.0010152258656818667,
            0.0027051050828588116,
            -0.0015501645601281979,
            0.008567787712297322,
            0.00028375156670324764,
            0.16533625857048248,
            32.45972485477316,
        )
        a0 = h55dot_opt[0]
        a1 = h55dot_opt[1]
        a2 = h55dot_opt[2]
        a3 = h55dot_opt[3]
        a4 = h55dot_opt[4]
        a5 = h55dot_opt[5]
        a6 = h55dot_opt[6]
        a7 = h55dot_opt[7]
        a8 = h55dot_opt[8]
        hdot_modes[5, 5] = (
            self.dm * (a0 + a1 * self.nu)
            + self.dm * (a2 + a3 * self.nu) * self.chi33
            + self.dm * (a4 + a5 * self.nu) * self.chi33**2
            + a6 * abs(self.dm * (a7 + a8 * self.nu) + self.chi33)
        )
        hdot_modes[3, 2] = (
            1.69342278665295764561 * self.nu**3
            - 0.14086998486546850606 * self.nu**2 * self.chi
            - 0.51099927320915217166 * self.nu**2
            - 0.01102715670910366617 * self.nu * self.chi**2
            + 0.0632109995418657783 * self.nu * self.chi
            + 0.02060709840137042725 * self.nu
            - 0.0008056890294201272 * self.chi**3
            + 0.00299932173169469322 * self.chi**2
            - 0.00678291732776604723 * self.chi
            + 0.0036736025144228808
        )

        h43dot_opt = (
            -0.0010672733086415792,
            0.012043361744214906,
            -0.004294799496844335,
            0.02224853600148107,
            0.0017733467790543556,
            -0.012158578417141666,
            0.0008201113714950076,
            3.880171117982288,
            -20.015435772594394,
        )
        a0 = h43dot_opt[0]
        a1 = h43dot_opt[1]
        a2 = h43dot_opt[2]
        a3 = h43dot_opt[3]
        a4 = h43dot_opt[4]
        a5 = h43dot_opt[5]
        a6 = h43dot_opt[6]
        a7 = h43dot_opt[7]
        a8 = h43dot_opt[8]
        hdot_modes[4, 3] = (
            self.dm * (a0 + a1 * self.nu)
            + self.dm * (a2 + a3 * self.nu) * self.chi33
            + self.dm * (a4 + a5 * self.nu) * self.chi33**2
            + a6 * abs(self.dm * (a7 + a8 * self.nu) + self.chi33)
        )
        return hdot_modes

    def hdotdot(self):
        """
        Fits for the waveform amplitude second derivative at the attachment point.
        See Appendix A3 of the [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of hdotdot values with keys being the desired mode
        """
        hdotdot_modes = {}
        hdotdot_modes[2, 2] = (
            -0.00335300225882774906 * self.nu**2
            + 0.00358851415965951012 * self.nu * self.chi
            - 0.00561520957901851664 * self.nu
            + 0.00038615328462788281 * self.chi**2
            + 0.00132564277249644174 * self.chi
            - 0.00245697909115198589
        )

        h33dotdot_opt = (
            0.0010293550284251136,
            0.0005517266548686501,
            -0.00021837584360501723,
            3.3961679011672516,
            32.212774869261885,
            -289.7723566230625,
            1331.981345413092,
            -2188.340922861859,
        )
        a0 = h33dotdot_opt[0]
        a1 = h33dotdot_opt[1]
        a2 = h33dotdot_opt[2]
        a3 = h33dotdot_opt[3]
        a4 = h33dotdot_opt[4]
        a5 = h33dotdot_opt[5]
        a6 = h33dotdot_opt[6]
        a7 = h33dotdot_opt[7]
        hdotdot_modes[3, 3] = self.dm * (a0 + a1 * self.nu) * self.chi33 + a2 * abs(
            self.dm
            * (
                a3
                + a4 * self.nu
                + a5 * self.nu**2
                + a6 * self.nu**3
                + a7 * self.nu**4
            )
            + self.chi33
        )
        h21dotdot_opt = (
            0.0001502578588400125,
            2.7742970362330567e-05,
            0.0036425847392171514,
            0.0005018530553996721,
            0.005681626755444207,
            -0.043290973710660874,
            -0.0003157923064414593,
            -0.0003722398103435137,
        )
        a0 = h21dotdot_opt[0]
        a1 = h21dotdot_opt[1]
        a2 = h21dotdot_opt[2]
        a3 = h21dotdot_opt[3]
        a4 = h21dotdot_opt[4]
        a5 = h21dotdot_opt[5]
        a6 = h21dotdot_opt[6]
        a7 = h21dotdot_opt[7]
        hdotdot_modes[2, 1] = a0 * self.dm - abs(
            self.dm * (a1 + a2 * self.nu)
            + self.dm * (a3 + a4 * self.nu + a5 * self.nu**2) * self.chi21D**2
            + a6 * self.chi21D**3
            + a7 * self.dm * self.chi21D
        )
        hdotdot_modes[4, 4] = (
            0.13849586076262568324 * self.nu**3
            - 0.0470083172660818796 * self.nu**2
            - 0.00059088846952625525 * self.nu * self.chi**2
            - 0.00050107688679214724 * self.nu * self.chi
            + 0.00389948502888598026 * self.nu
            + 0.00017443998182683801 * self.chi**2
            + 0.00031810951384450125 * self.chi
            - 0.00045065044351543849
        )

        h55dotdot_opt = (
            0.0001177890815495443,
            -5.931974803382094e-05,
            -6.787407227964159e-05,
            0.00024603419335046537,
            -5.5520548080207735e-05,
            0.00027766074728060057,
        )
        a0 = h55dotdot_opt[0]
        a1 = h55dotdot_opt[1]
        a2 = h55dotdot_opt[2]
        a3 = h55dotdot_opt[3]
        a4 = h55dotdot_opt[4]
        a5 = h55dotdot_opt[5]
        hdotdot_modes[5, 5] = (
            self.dm * (a0 + a1 * self.nu)
            + self.dm * (a2 + a3 * self.nu) * self.chi33
            + (a4 + a5 * self.nu) * self.chi33**2
        )
        hdotdot_modes[3, 2] = (
            0.20836028051484326018 * self.nu**3
            - 0.02746063216748286309 * self.nu**2 * self.chi
            - 0.05319145260701527156 * self.nu**2
            - 0.00288235478694781307 * self.nu * self.chi**2
            + 0.00848086944391238627 * self.nu * self.chi
            + 0.00160402472655379815 * self.nu
            + 0.00070714933953842658 * self.chi**2
            - 0.0006905793194749726 * self.chi
            - 5.641060229726973e-5
        )

        h43dotdot_opt = (
            -0.00034837066035625267,
            0.0029095133385644,
            -4.886585508152058e-06,
            23.849356773565383,
            -531.9652630324141,
            291.7510533819259,
            12647.805786829585,
            -25646.358742128104,
        )
        a0 = h43dotdot_opt[0]
        a1 = h43dotdot_opt[1]
        a2 = h43dotdot_opt[2]
        a3 = h43dotdot_opt[3]
        a4 = h43dotdot_opt[4]
        a5 = h43dotdot_opt[5]
        a6 = h43dotdot_opt[6]
        a7 = h43dotdot_opt[7]
        hdotdot_modes[4, 3] = self.dm * (a0 + a1 * self.nu) * self.chi33 + a2 * abs(
            self.dm
            * (
                a3
                + a4 * self.nu
                + a5 * self.nu**2
                + a6 * self.nu**3
                + a7 * self.nu**4
            )
            + self.chi33
        )
        return hdotdot_modes

    def omega(self):
        """
        Fits for the waveform frequency at the attachment point.
        See Appendix A4 of the [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of omega values with keys being the desired mode
        """
        omega_modes = {}
        omega_modes[2, 2] = (
            5.89352329617707670906 * self.nu**4
            + 3.75145580491965446868 * self.nu**3 * self.chi
            - 3.34930536209472551334 * self.nu**3
            - 0.97140932625194231775 * self.nu**2 * self.chi**2
            - 1.69734302394369973577 * self.nu**2 * self.chi
            + 0.28539204856044564362 * self.nu**2
            + 0.2419483723662931296 * self.nu * self.chi**3
            + 0.51801427018052081941 * self.nu * self.chi**2
            + 0.25096450064948544467 * self.nu * self.chi
            - 0.31709602351033533418 * self.nu
            - 0.01525897668158244028 * self.chi**4
            - 0.06692658483513916345 * self.chi**3
            - 0.08715176045684569495 * self.chi**2
            - 0.09133931944098934441 * self.chi
            - 0.2685414392185025978
        )
        omega_modes[3, 3] = (
            8.88716288087403682994 * self.nu**3
            - 0.74592378229749944918 * self.nu**2 * self.chi
            - 4.22683134159469808822 * self.nu**2
            + 0.34667482329409210484 * self.nu * self.chi**2
            + 0.4789145868044565324 * self.nu * self.chi
            - 0.04514088822512542926 * self.chi**3
            - 0.11941861972305316264 * self.chi**2
            - 0.17467027880037502841 * self.chi
            - 0.42716659841853366064
        )
        omega_modes[2, 1] = (
            -1.96515745599907942776 * self.nu**3
            - 0.16885361554827885144 * self.nu**2 * self.chi
            + 0.53085041746959926723 * self.nu**2
            + 0.07734281111269736275 * self.nu * self.chi**2
            + 0.15938183067714695174 * self.nu * self.chi
            - 0.2379037879767669228 * self.nu
            - 0.01009016749505482584 * self.chi**3
            - 0.02410957086017969861 * self.chi**2
            - 0.04763491663090872047 * self.chi
            - 0.17652608750683201899
        )
        omega_modes[4, 4] = (
            13.65133489176671410803 * self.nu**3
            - 0.76871184038256423765 * self.nu**2 * self.chi
            - 5.49032933904132924852 * self.nu**2
            + 0.4158639192986712807 * self.nu * self.chi**2
            + 0.5925675710398274898 * self.nu * self.chi
            - 0.04252924146785119763 * self.chi**3
            - 0.1552216774031633939 * self.chi**2
            - 0.24450814728556427569 * self.chi
            - 0.57404100593064200098
        )
        omega_modes[5, 5] = (
            13.81386002062406426205 * self.nu**3
            - 3.0457597325553762424 * self.nu**2 * self.chi
            - 6.61611019771898423159 * self.nu**2
            + 0.80275927256688528466 * self.nu * self.chi**2
            + 1.43470995594104122617 * self.nu * self.chi
            + 0.47247403131411785937 * self.nu
            - 0.09162925213075370778 * self.chi**3
            - 0.24664558041625297968 * self.chi**2
            - 0.32959068465244367729 * self.chi
            - 0.58934113264394660803
        )
        omega_modes[3, 2] = (
            -2.34602394381327350459 * self.nu**3
            - 2.75863454206899216814 * self.nu**2 * self.chi
            + 1.57985993022275095221 * self.nu**2
            + 0.81135314747739029073 * self.nu * self.chi
            - 0.31775591479035469877 * self.nu
            - 0.04564745621234577583 * self.chi**2
            - 0.11247674297883318573 * self.chi
            - 0.33114128825887062524
        )
        omega_modes[4, 3] = (
            -55.53410548228862353426 * self.nu**3
            - 0.90591912935891649727 * self.nu**2 * self.chi
            + 23.91327719774037419143 * self.nu**2
            + 0.22690304977596567615 * self.nu * self.chi**2
            + 0.29109174621004529904 * self.nu * self.chi
            - 3.48798589848595197438 * self.nu
            - 0.0379192517897683698 * self.chi**3
            - 0.0872880306983466886 * self.chi**2
            - 0.11979971300165934145 * self.chi
            - 0.3430602590276149999
        )
        return omega_modes

    def omegadot(self):
        """
        Fits for the waveform frequency first derivative at the attachment point.
        See Appendix A4 of the [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of omegadot values with keys being the desired mode
        """
        omegadot_modes = {}
        omegadot_modes[2, 2] = (
            -0.23712612963269574795 * self.nu**3
            + 0.07799016321986182443 * self.nu**2 * self.chi
            + 0.09221479145462828375 * self.nu**2
            - 0.00839285104015016943 * self.nu * self.chi**2
            - 0.02877175649350346628 * self.nu * self.chi
            - 0.03103970973029888253 * self.nu
            + 0.00061394267373741083 * self.chi**3
            + 0.0019481328417233967 * self.chi**2
            + 0.0017051416772119448 * self.chi
            - 0.0054839528158373476
        )
        omegadot_modes[3, 3] = (
            0.2554015440024767214 * self.nu**3
            + 0.1543776301642514337 * self.nu**2 * self.chi
            - 0.0866295463333640603 * self.nu**2
            - 0.01623076677816887373 * self.nu * self.chi**2
            - 0.05061758919542676954 * self.nu * self.chi
            - 0.02740501185261355263 * self.nu
            + 0.00169663712021324796 * self.chi**3
            + 0.00398506811959840803 * self.chi**2
            + 0.00272137645449461703 * self.chi
            - 0.00973636029227470737
        )
        omegadot_modes[2, 1] = (
            -0.20436806016561828714 * self.nu**3
            + 0.03383058374774079724 * self.nu**2 * self.chi
            + 0.12070472224065939559 * self.nu**2
            - 0.00896529609487783756 * self.nu * self.chi**2
            - 0.00575216782379691424 * self.nu * self.chi
            - 0.03514372134290563721 * self.nu
            + 0.0014901043978378613 * self.chi**3
            + 0.00273944957648775004 * self.chi**2
            + 0.00200277469830409133 * self.chi
            - 0.00657904072987968228
        )
        omegadot_modes[4, 4] = (
            0.53666427603875110908 * self.nu**3
            + 0.16269258337509870382 * self.nu**2 * self.chi
            - 0.09479659696815241621 * self.nu**2
            - 0.02468745449256098956 * self.nu * self.chi**2
            - 0.06120499833132028722 * self.nu * self.chi
            - 0.04540599000644451183 * self.nu
            + 0.00181212931541878787 * self.chi**3
            + 0.00568015431535457815 * self.chi**2
            + 0.00362281246962605202 * self.chi
            - 0.01303752234990177353
        )
        omegadot_modes[5, 5] = (
            0.04396263632039092845 * self.nu**3
            + 0.16401057342208869017 * self.nu**2 * self.chi
            + 0.0480445690111714549 * self.nu**2
            - 0.01547038929169576409 * self.nu * self.chi**2
            - 0.05651624188739841348 * self.nu * self.chi
            - 0.04519696852035513107 * self.nu
            + 0.00150884009977045211 * self.chi**3
            + 0.00280221939154576445 * self.chi**2
            + 0.00207193006850621116 * self.chi
            - 0.00868773813796784242
        )
        omegadot_modes[3, 2] = (
            -2.49478784743875880991 * self.nu**3
            + 0.09192003793052351546 * self.nu**2 * self.chi
            + 0.99511595542597619524 * self.nu**2
            - 0.03671064886430309981 * self.nu * self.chi**2
            - 0.03071335466752724355 * self.nu * self.chi
            - 0.10162991709458844836 * self.nu
            + 0.00553205070234718318 * self.chi**2
            + 0.0059265967880560216 * self.chi
            - 0.0107628728873452352
        )
        omegadot_modes[4, 3] = (
            -5.16061307106004907297 * self.nu**3
            + 0.1329597836037081926 * self.nu**2 * self.chi
            + 2.18078055419966032602 * self.nu**2
            - 0.00987590957480824309 * self.nu * self.chi**2
            - 0.06088357656881192986 * self.nu * self.chi
            - 0.29260675353893633721 * self.nu
            + 0.00053653086065616224 * self.chi**3
            + 0.00327866342069590847 * self.chi**2
            + 0.00851322039320196768 * self.chi
            - 0.00530816071704910165
        )
        return omegadot_modes
