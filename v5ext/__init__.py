"""V5 extensions: microstructure, sentiment, ML, Bayesian scoring, backtesting, portfolio, optimizer.

كل الموديولات هنا مستقلة عن الماسح، والماسح بيستدعيها عبر live_bridge فقط.
أي عطل في أي موديول يرجّع قيمة محايدة (None) ولا يوقف دورة المسح.
"""
__version__ = "5.1.0"
