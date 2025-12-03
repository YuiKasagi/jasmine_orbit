from datetime import datetime, timezone, timedelta


def setdate(args):
    # args : (-s|-a) -p <day_offset> -w <days>
    # NAOJ によれば 日本では
    #  year      春分           秋分
    #  2030年  3月20日(水)  9月23日(月)
    #  2031年  3月21日(金)  9月23日(火)
    #  2032年  3月20日(土)  9月22日(水)
    #  2033年  3月20日(日)  9月23日(金)
    #  2034年  3月20日(月)  9月23日(土)
    #  2035年  3月21日(水)  9月23日(日)
    jst = timezone(timedelta(hours=9))
    if args['-a']:  # autumn 秋 JST(UTC+9)で与える
        ref_date_jst = datetime(2030, 9, 23, 0, 0, 0, tzinfo=jst)
    elif args['-s']:  # sprint 春 JST(UTC+9)で与える
        ref_date_jst = datetime(2030, 3, 20, 0, 0, 0, tzinfo=jst)
    ref_date = ref_date_jst.astimezone(timezone.utc)
    print('基準日 {} (UTC) '.format(ref_date.strftime("%Y/%m/%d %H:%M:%S")))
    # 与えられた期間のシミュレーション
    day_offset = float(args['-p'])
    days_calc = float(args['-w'])
    start_date = ref_date + timedelta(days=day_offset)
    print('開始日 {} (UTC)'.format(start_date.strftime("%Y/%m/%d %H:%M:%S")))
    print('計算時間 {} days'.format(days_calc))
    end_date = start_date + timedelta(days=days_calc)

    return start_date, days_calc, end_date

def get_refdate(year=2030):
    jst = timezone(timedelta(hours=9))
    if year==2030:
        ref_date_autumn_jst = datetime(2030, 9, 23, 0, 0, 0, tzinfo=jst)
        ref_date_spring_jst = datetime(2030, 3, 20, 0, 0, 0, tzinfo=jst)
    ref_date_autumn = ref_date_autumn_jst.astimezone(timezone.utc)
    ref_date_spring = ref_date_spring_jst.astimezone(timezone.utc)
    return ref_date_spring, ref_date_autumn

def calc_obliquity(julian_date):
    """calculate obliquity 

        Args:
            julian_date: julian date
        Returns:
            Earth's axis tilt in degree (IAU)
            ref. Capitaine, Wallace and Chapront, 2003, A&A, 412, 567
    """
    julian_centuries = (julian_date - 2451545.0) / 36525.0

    epsilon_zero = 84381.448  # obliquity at J2000.0
    epsilon = epsilon_zero \
                - 46.8150 * julian_centuries \
                - 0.00059 * julian_centuries**2 \
                + 0.001813 * julian_centuries**2 
    return epsilon / 3600.
    