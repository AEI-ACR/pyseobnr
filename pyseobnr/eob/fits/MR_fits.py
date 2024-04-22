"""
Contains fits for coefficients in the merger-ringdown ansatze for all modes.
"""

import numpy as np


class MergerRingdownFits:
    """
    Contains fits for coefficients in the merger-ringdown ansatze for all modes.

    Class that wraps all necessary fits as methods. Each method returns a dict with
    keys being the desired mode.

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

    def c1f(self):
        """
        Fits for the c1f coefficient entering the merger-ringdown amplitude ansatz.
        See Eq(57) and Appendix B of [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of c1f values with keys being the desired mode
        """
        c1f_modes = {}

        c1f_modes[2, 2] = (
            -4.23824640099272276217 * self.nu**4
            + 1.86797765630645606905 * self.nu**3 * self.chi
            + 2.04371156181773017124 * self.nu**3
            + 0.01316051994812337048 * self.nu**2 * self.chi**2
            - 0.70248812998796750229 * self.nu**2 * self.chi
            - 0.40699224653253718298 * self.nu**2
            + 0.06284152952186422558 * self.nu * self.chi**3
            + 0.04938791239730106614 * self.nu * self.chi**2
            + 0.03388527758147390212 * self.nu * self.chi
            + 0.05358902726316702725 * self.nu
            - 0.00177674004113037185 * self.chi**4
            - 0.01890815113487190682 * self.chi**3
            - 0.01931426101231131093 * self.chi**2
            - 0.01161150126773277842 * self.chi
            + 0.08625435880606084627
        )
        c1f_modes[3, 3] = (
            -5.84741730773623924478 * self.nu**3
            - 0.49452398211101195047 * self.nu**2 * self.chi
            + 1.95746216874706702171 * self.nu**2
            + 0.02945901056990525221 * self.nu * self.chi**2
            + 0.16946282369268209078 * self.nu * self.chi
            - 0.17168187932239820093 * self.nu
            - 0.00955973829289496688 * self.chi**3
            - 0.02026440669176455753 * self.chi**2
            - 0.02628548755716426585 * self.chi
            + 0.09353927839792018639
        )
        c1f_modes[2, 1] = (
            -2.93473514640184562907 * self.nu**3
            + 0.19746688649224189427 * self.nu**2 * self.chi
            + 1.00910612882264860701 * self.nu**2
            + 0.17346184748365778283 * self.nu * self.chi**2
            - 0.11272119541174477342 * self.nu
            - 0.02887336883447702562 * self.chi**2
            - 0.0261389677638197114 * self.chi
            + 0.09988905108470733096
        )
        c1f_modes[4, 4] = (
            -1656.06543875322722669807 * self.nu**4
            + 817.8357264462189277765 * self.nu**3
            + 4.51950362264643246135 * self.nu**2 * self.chi
            - 127.05537902659793303428 * self.nu**2
            - 1.48903628096446793982 * self.nu * self.chi
            + 6.92196765840723760022 * self.nu
            + 0.06840293918557056874 * self.chi
            + 0.00938647985547616526
        )
        c1f_modes[5, 5] = (
            -5.96689105456452928422 * self.nu**3
            - 0.20681139053047220555 * self.nu**2 * self.chi
            + 1.76927984136387994596 * self.nu**2
            + 0.05974788269195114437 * self.nu * self.chi**2
            + 0.05507770250798054573 * self.nu * self.chi
            - 0.05527152193316976075 * self.nu
            - 0.00995731903745558297 * self.chi**3
            - 0.02145967586159740551 * self.chi**2
            - 0.01452795510255427969 * self.chi
            + 0.08036815955900285247
        )
        c1f_modes[3, 2] = (
            -26.99180579320305994884 * self.nu**3
            + 8.98776310541993694869 * self.nu**2 * self.chi
            + 13.71680055048422630648 * self.nu**2
            + 0.64168103283614175147 * self.nu * self.chi**2
            - 1.5822588157563266531 * self.nu * self.chi
            - 1.63082958579053993731 * self.nu
            - 0.13303531460644799078 * self.chi**3
            - 0.11186538668494711013 * self.chi**2
            + 0.09560378719809306536 * self.chi
            + 0.1575429834375732252
        )
        c1f_modes[4, 3] = (
            44.31194766851576360978 * self.nu**3
            + 4.18890790614698360628 * self.nu**2 * self.chi
            - 22.114177403007090561 * self.nu**2
            - 1.36573150894960360091 * self.nu * self.chi
            + 3.38608198540108329411 * self.nu
            + 0.0415847487543668029 * self.chi**3
            + 0.05890807806783741885 * self.chi
            - 0.0353145076510041761
        )

        return c1f_modes

    def c2f(self):
        """
        Fits for the c2f coefficient entering the merger-ringdown amplitude ansatz.
        See Eq(57) and Appendix B of [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of c2f values with keys being the desired mode
        """
        c2f_modes = {}
        c2f_modes[2, 2] = (
            -63.28645899089733006804 * self.nu**4
            + 2.00294725303467924249 * self.nu**3 * self.chi
            + 44.33138899436394098075 * self.nu**3
            - 3.55617293922388588712 * self.nu**2 * self.chi**2
            - 5.58585057654383287939 * self.nu**2 * self.chi
            - 9.5295732728313318205 * self.nu**2
            + 1.02187518454288950309 * self.nu * self.chi**3
            + 1.97008188121834493245 * self.nu * self.chi**2
            + 1.83772448389004638969 * self.nu * self.chi
            + 1.15569522525235401922 * self.nu
            - 0.20348032514327910047 * self.chi**3
            - 0.2642970192733161694 * self.chi**2
            - 0.27076037187561419195 * self.chi
            - 0.52876279548305116229
        )
        c2f_modes[3, 3] = (
            14.80191571837246300447 * self.nu**3
            - 4.2506085799341262188 * self.nu**2 * self.chi
            - 7.06058144862581293921 * self.nu**2
            + 0.23710749614144122077 * self.nu * self.chi**2
            + 1.7631048513797487054 * self.nu * self.chi
            + 1.1586271594235801885 * self.nu
            - 0.05734634823328856046 * self.chi**3
            - 0.09428487098033955238 * self.chi**2
            - 0.31582567372407910344 * self.chi
            - 0.64688786077920956696
        )
        c2f_modes[2, 1] = (
            -54.91758465160924629345 * self.nu**3
            - 20.79282498761616082561 * self.nu**2 * self.chi
            + 16.46631184405826431316 * self.nu**2
            + 6.86774560726690541657 * self.nu * self.chi
            + 0.42631595523344306686 * self.nu
            + 0.18348940531384577701 * self.chi**3
            + 0.10572967980814534927 * self.chi**2
            - 0.48494791368293987954 * self.chi
            - 0.92208038677972470332
        )
        c2f_modes[4, 4] = (
            -393.72770177118201218036 * self.nu**4
            - 254.71955194900522201351 * self.nu**3 * self.chi
            + 145.32787996648792727683 * self.nu**3
            - 12.64781420820250623649 * self.nu**2 * self.chi**2
            + 105.69879137093592191832 * self.nu**2 * self.chi
            - 15.55622197261752681641 * self.nu**2
            + 0.96486143874433238921 * self.nu * self.chi**3
            + 5.26496904337297255694 * self.nu * self.chi**2
            - 12.10728120729313239679 * self.nu * self.chi
            + 1.59244900534151656579 * self.nu
            - 0.18522585518375980773 * self.chi**3
            - 0.53972090630942404221 * self.chi**2
            + 0.22439952081335490242 * self.chi
            - 0.67766429745998968404
        )
        c2f_modes[5, 5] = (
            -1887.59110187702754046768 * self.nu**4
            - 28.49927819685526841909 * self.nu**3 * self.chi
            + 794.1347113664942298783 * self.nu**3
            + 1.6383448468383090546 * self.nu**2 * self.chi**2
            + 3.73033983487164366721 * self.nu**2 * self.chi
            - 107.010824091130814395 * self.nu**2
            + 1.85372256088708575739 * self.nu * self.chi
            + 6.32116957997830386518 * self.nu
            + 0.11970254052165903158 * self.chi**4
            - 0.06472473552861608692 * self.chi**2
            - 0.22528339413615969256 * self.chi
            - 1.50748281609513234969
        )
        c2f_modes[3, 2] = (
            -51.46977309098925701392 * self.nu**3
            - 25.54493144533089221682 * self.nu**2 * self.chi
            + 46.20983330953832535215 * self.nu**2
            - 1.59062297059173762825 * self.nu * self.chi**2
            + 10.12796791115255423676 * self.nu * self.chi
            - 6.48457068880731757332 * self.nu
            + 0.12160774521877980303 * self.chi**3
            + 0.16723081635954289981 * self.chi**2
            - 0.99906205283978699594 * self.chi
            - 0.71688333369378942628
        )
        c2f_modes[4, 3] = (
            -57.96882060761311805663 * self.nu**3
            - 9.80318668122179559532 * self.nu**2 * self.chi
            + 7.82092906762141826249 * self.nu**2
            + 0.33723535549437705372 * self.nu * self.chi**2
            + 3.9951985799433136215 * self.nu * self.chi
            + 3.36474064193650734822 * self.nu
            + 0.12576369978108004055 * self.chi**3
            + 0.14620175946137692335 * self.chi**2
            - 0.24097631369535654766 * self.chi
            - 1.1217157841552500841
        )

        return c2f_modes

    def d1f(self):
        """
        Fits for the d1f coefficient entering the merger-ringdown phase ansatz.
        See Eq(58) and Appendix B of [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of c2f values with keys being the desired mode
        """
        d1f_modes = {}
        d1f_modes[2, 2] = (
            -28.42370101139921700906 * self.nu**4
            + 4.11346289839689127632 * self.nu**3 * self.chi
            + 20.71987362022024470321 * self.nu**3
            + 1.03335215030655280799 * self.nu**2 * self.chi**2
            - 1.65292430358775521704 * self.nu**2 * self.chi
            - 6.07567868511363951001 * self.nu**2
            + 0.04730524488221983515 * self.nu * self.chi**3
            - 0.254350860993373451 * self.nu * self.chi**2
            + 0.09083410987717309426 * self.nu * self.chi
            + 0.78009259453928059269 * self.nu
            - 0.01332056979451640664 * self.chi**4
            - 0.0242033556149483034 * self.chi**3
            - 0.00784682245346276369 * self.chi**2
            + 0.1357578010277912528
        )
        d1f_modes[3, 3] = (
            2.31643418192078076601 * self.nu**3
            + 0.67844229400588917933 * self.nu**2 * self.chi
            - 2.19222695489746532971 * self.nu**2
            + 0.22146624271406206708 * self.nu * self.chi**2
            - 0.26126441061361949103 * self.nu * self.chi
            + 0.42458214052259168891 * self.nu
            - 0.01652448770503477798 * self.chi**3
            - 0.066322821567381951 * self.chi**2
            + 0.0066644906887992272 * self.chi
            + 0.16157693147048388105
        )
        d1f_modes[2, 1] = (
            42.71553360276344335489 * self.nu**4
            - 10.64852642435220175798 * self.nu**3 * self.chi
            - 18.28060258075289112867 * self.nu**3
            - 0.87720079740200784801 * self.nu**2 * self.chi**2
            + 4.10491801013824897382 * self.nu**2 * self.chi
            + 2.23659207909034307704 * self.nu**2
            + 0.39862147091294380941 * self.nu * self.chi**3
            + 0.41455285041784462052 * self.nu * self.chi**2
            - 0.72357614572043449375 * self.nu * self.chi
            - 0.04809402092838542531 * self.nu
            + 0.01846697992790376913 * self.chi**4
            - 0.05049883412734225419 * self.chi**3
            - 0.06827724811631767643 * self.chi**2
            + 0.0392267301677774044 * self.chi
            + 0.16335041746961706521
        )
        d1f_modes[4, 4] = (
            -44.2808149959025172393 * self.nu**3
            + 4.29798528136823421164 * self.nu**2 * self.chi
            + 11.02148212846928743147 * self.nu**2
            + 0.49422079003244095974 * self.nu * self.chi**2
            - 1.28438557119347906976 * self.nu * self.chi
            - 0.16294342992143145965 * self.nu
            - 0.0206442049610057049 * self.chi**3
            - 0.12707398376969430975 * self.chi**2
            + 0.06268399269769617255 * self.chi
            + 0.16601792669109360912
        )
        d1f_modes[5, 5] = (
            9.01854564140440295716 * self.nu**3
            + 0.87179891034374046299 * self.nu**2 * self.chi
            - 5.00948762747145970309 * self.nu**2
            + 0.16807146230986744206 * self.nu * self.chi**2
            - 0.23005664661285465944 * self.nu * self.chi
            + 0.6063125875447843427 * self.nu
            - 0.02153736690702941908 * self.chi**3
            - 0.05026259943312909317 * self.chi**2
            + 0.15062218133224089534
        )
        d1f_modes[3, 2] = np.exp(
            81.22217466619477477252 * self.nu**3
            - 0.51829138617583514481 * self.nu**2 * self.chi
            - 18.0405291714101458922 * self.nu**2
            - 8.68472246789598223415 * self.nu * self.chi**2
            - 1.40793405689229178535 * self.nu * self.chi
            + 2.21640625816665304271 * self.nu
            - 0.7640149698849937332 * self.chi**3
            + 0.69194585915884021521 * self.chi**2
            + 0.23642724410018603476 * self.chi
            - 1.87945499992947073764
        )
        d1f_modes[4, 3] = np.exp(
            366.64564526161473168031 * self.nu**3
            - 14.82339076176638137383 * self.nu**2 * self.chi
            - 161.73251327983666669752 * self.nu**2
            + 3.9786897816820756546 * self.nu * self.chi**2
            + 6.94085604691350077644 * self.nu * self.chi
            + 19.56469878860015754185 * self.nu
            - 0.88828586730455139087 * self.chi**3
            - 1.04718112449004019382 * self.chi**2
            - 0.36780123953143395443 * self.chi
            - 2.29578006968414349842
        )

        return d1f_modes

    def d2f(self):
        """
        Fits for the d2f coefficient entering the merger-ringdown phase ansatz.
        See Eq(58) and Appendix B of [SEOBNRv5HM-notes]_ .

        Returns:
            dict: dictionary of c2f values with keys being the desired mode
        """
        d2f_modes = {}
        d2f_modes[2, 2] = np.exp(
            -352.24938296898454836992 * self.nu**4
            + 9.05730635731021394008 * self.nu**3 * self.chi
            + 275.84349920209979245556 * self.nu**3
            + 23.975132253988164166 * self.nu**2 * self.chi**2
            - 5.26829618908132601973 * self.nu**2 * self.chi
            - 81.48331396357356481985 * self.nu**2
            - 3.39885766491276442025 * self.nu * self.chi**3
            - 10.06495407151063048445 * self.nu * self.chi**2
            + 0.46455322692280037744 * self.nu * self.chi
            + 11.18457585889310479388 * self.nu
            - 0.1631129108825159213 * self.chi**4
            + 0.728816370357556087 * self.chi**3
            + 1.2114999080794128794 * self.chi**2
            + 0.56269034372727599891 * self.chi
            + 0.03570970180918431325
        )
        d2f_modes[3, 3] = np.exp(
            324.31022294062370292522 * self.nu**3
            + 29.07251486043331212272 * self.nu**2 * self.chi
            - 124.68188129288633092528 * self.nu**2
            - 1.83069476195595881585 * self.nu * self.chi**2
            - 10.58131925489262670226 * self.nu * self.chi
            + 13.20042565939421663757 * self.nu
            + 0.27599881348329013964 * self.chi**3
            + 0.51273402705147774761 * self.chi**2
            + 1.31064330152030117382 * self.chi
            + 0.41085534754131985968
        )
        d2f_modes[2, 1] = np.exp(
            91.26918293911697332987 * self.nu**3
            + 6.44667034904012936636 * self.nu**2 * self.chi
            - 27.3297506838912305227 * self.nu**2
            - 1.19736283448519631456 * self.nu * self.chi**2
            - 5.63056276865802374232 * self.nu * self.chi
            + 1.10126233822357844083 * self.nu
            + 0.81408505221017091191 * self.chi**3
            + 0.56062237457558927733 * self.chi**2
            + 0.94958568844285495825 * self.chi
            + 1.04076128273052770368
        )
        d2f_modes[4, 4] = np.exp(
            -528.36891537776148197736 * self.nu**3
            + 37.73511579834331541861 * self.nu**2 * self.chi
            + 155.11519575456409825165 * self.nu**2
            - 12.51666897674352618708 * self.nu * self.chi
            - 6.61244848534673668183 * self.nu
            + 1.30986803286068065333 * self.chi
            + 0.78772617222401941639
        )
        d2f_modes[5, 5] = np.exp(
            143.74520772547040792233 * self.nu**3
            + 28.83903496709819336274 * self.nu**2 * self.chi
            - 64.47822669258478356369 * self.nu**2
            - 9.72602503470527857132 * self.nu * self.chi
            + 6.22383279275213752157 * self.nu
            + 0.90142275047914954822 * self.chi
            + 2.05813880795029913173
        )
        d2f_modes[3, 2] = np.exp(
            -215.37239883246164140473 * self.nu**3
            - 39.32457858880395917822 * self.nu**2 * self.chi
            + 136.2093599792216878086 * self.nu**2
            - 24.50150326852915583231 * self.nu * self.chi**2
            + 14.37990077612682426889 * self.nu * self.chi
            - 16.84281601681732354336 * self.nu
            - 1.8198216679619081404 * self.chi**3
            + 3.28788232667572932755 * self.chi**2
            + 1.46348537186546634459
        )
        d2f_modes[4, 3] = np.exp(
            730.42295952719962315314 * self.nu**3
            + 39.21795990563291667286 * self.nu**2 * self.chi
            - 312.96059813044456632269 * self.nu**2
            - 10.6511668077322863013 * self.nu * self.chi
            + 37.40256671810621469376 * self.nu
            - 0.95067622352941993924 * self.chi**3
            - 0.31428040326336281751 * self.chi**2
            + 1.33973212065423852302 * self.chi
            - 0.06189359885595324684
        )

        return d2f_modes
