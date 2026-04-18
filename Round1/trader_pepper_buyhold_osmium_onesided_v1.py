
from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder
from typing import Dict, List, Any, Optional, Tuple
import json
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.maxLogLength = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: Dict[str, List[Order]],
        conversions: int,
        traderData: str,
    ) -> None:
        baseLength = len(
            self.toJson(
                [
                    self.compressState(state, ""),
                    self.compressOrders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        maxItemLength = max(0, (self.maxLogLength - baseLength) // 3)

        print(
            self.toJson(
                [
                    self.compressState(state, self.truncate(state.traderData, maxItemLength)),
                    self.compressOrders(orders),
                    conversions,
                    self.truncate(traderData, maxItemLength),
                    self.truncate(self.logs, maxItemLength),
                ]
            )
        )

        self.logs = ""

    def compressState(self, state: TradingState, traderData: str):
        return [
            state.timestamp,
            traderData,
            self.compressListings(state.listings),
            self.compressOrderDepths(state.order_depths),
            self.compressTrades(state.own_trades),
            self.compressTrades(state.market_trades),
            state.position,
            self.compressObservations(state.observations),
        ]

    def compressListings(self, listings):
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compressOrderDepths(self, orderDepths):
        compressed = {}
        for symbol, orderDepth in orderDepths.items():
            compressed[symbol] = [orderDepth.buy_orders, orderDepth.sell_orders]
        return compressed

    def compressTrades(self, trades):
        compressed = []
        for tradeList in trades.values():
            for trade in tradeList:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compressObservations(self, observations):
        conversionObservations = {}
        if hasattr(observations, "conversionObservations"):
            for product, observation in observations.conversionObservations.items():
                conversionObservations[product] = [
                    observation.bidPrice,
                    observation.askPrice,
                    observation.transportFees,
                    observation.exportTariff,
                    observation.importTariff,
                    observation.sunlight,
                    observation.humidity,
                ]

        plainValueObservations = {}
        if hasattr(observations, "plainValueObservations"):
            plainValueObservations = observations.plainValueObservations

        return [plainValueObservations, conversionObservations]

    def compressOrders(self, orders: Dict[str, List[Order]]):
        compressed = []
        for orderList in orders.values():
            for order in orderList:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def toJson(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, maxLength: int) -> str:
        if len(value) <= maxLength:
            return value
        return value[: maxLength - 3] + "..."


logger = Logger()


class Product:
    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"


PARAMS: Dict[str, Dict[str, float]] = {
    Product.PEPPER: {
        "position_limit": 80,
        "regression_window": 100,
        "history_keep": 220,
        "residual_window": 120,
        "min_residual_std": 1.5,
        "take_edge": 4.0,
        "quote_edge": 3.0,
        "join_threshold": 6.0,
        "base_order_size": 10,
        "soft_position_limit": 45,
        "inventory_skew_per_10": 1.0,
        "clear_edge": 1.0,
        "target_early": 80,
        "target_mid": 80,
        "target_late": 70,
        "early_end_ts": 250000,
        "mid_end_ts": 750000,
        "late_reduce_ts": 930000,
        "hold_premium_early": 12.0,
        "hold_premium_mid": 8.0,
        "hold_premium_late": 4.0,
        "rich_sell_edge": 16.0,
    },
    Product.OSMIUM: {
        "position_limit": 80,
        "mean_window": 20,
        "history_keep": 50,
        "take_edge": 2.5,
        "quote_edge": 3.0,
        "join_threshold": 7.0,
        "base_order_size": 12,
        "soft_position_limit": 24,
        "inventory_skew_per_8": 1.25,
        "clear_edge": 0.5,
        "imbalance_coeff": 1.0,
        "last_return_coeff": 0.35,
        "momentum5_coeff": 0.08,
    },
}


class Trader:
    POSITION_LIMITS = {
        Product.PEPPER: 80,
        Product.OSMIUM: 80,
    }
    DEBUG_EVERY = 250
    DEBUG_POSITION = 30

    def __init__(self) -> None:
        self.debugProducts = set()

    def shouldLogProduct(self, product: str) -> bool:
        return product in self.debugProducts

    def run(self, state: TradingState):
        traderObject = self.loadTraderData(state.traderData)
        result: Dict[str, List[Order]] = {}
        conversions = 0

        periodicLog = (state.timestamp % self.DEBUG_EVERY == 0)
        if periodicLog:
            logger.print("timestamp =", state.timestamp)
            logger.print("positions =", state.position)

        self.debugProducts = set()

        for product in [Product.PEPPER, Product.OSMIUM]:
            if product not in state.order_depths:
                continue

            orderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            ownTrades = self.summarizeOwnTrades(state, product)

            if periodicLog or abs(position) >= self.DEBUG_POSITION or len(ownTrades) > 0:
                self.debugProducts.add(product)

            bestBid = self.getBestBid(orderDepth)
            bestAsk = self.getBestAsk(orderDepth)
            spread = None if bestBid is None or bestAsk is None else bestAsk - bestBid

            if self.shouldLogProduct(product):
                logger.print(
                    product,
                    "book",
                    "bb =", bestBid,
                    "ba =", bestAsk,
                    "spread =", spread,
                    "topBids =", self.formatLevels(self.sortedBids(orderDepth), 2),
                    "topAsks =", self.formatLevels(self.sortedAsks(orderDepth), 2),
                    "pos =", position,
                    "ownTrades =", ownTrades,
                )

            if product == Product.PEPPER:
                signal = self.updatePepperFairValue(traderObject, product, orderDepth)
                orders = self.tradePepper(product, orderDepth, signal, position, state.timestamp)
            else:
                signal = self.updateOsmiumFairValue(traderObject, product, orderDepth)
                orders = self.tradeOsmium(product, orderDepth, signal, position)

            result[product] = orders

        traderData = json.dumps(traderObject, separators=(",", ":"))
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def loadTraderData(self, traderData: str) -> dict:
        if traderData is None or traderData == "":
            return {}
        try:
            return json.loads(traderData)
        except Exception:
            return {}

    def getBestAsk(self, orderDepth: OrderDepth) -> Optional[int]:
        if not orderDepth.sell_orders:
            return None
        return min(orderDepth.sell_orders.keys())

    def getBestBid(self, orderDepth: OrderDepth) -> Optional[int]:
        if not orderDepth.buy_orders:
            return None
        return max(orderDepth.buy_orders.keys())

    def getMidPrice(self, orderDepth: OrderDepth) -> Optional[float]:
        bestBid = self.getBestBid(orderDepth)
        bestAsk = self.getBestAsk(orderDepth)
        if bestBid is None or bestAsk is None:
            return None
        return (bestBid + bestAsk) / 2.0

    def sortedAsks(self, orderDepth: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(orderDepth.sell_orders.items())

    def sortedBids(self, orderDepth: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(orderDepth.buy_orders.items(), reverse=True)

    def appendHistory(self, traderObject: dict, key: str, value: float, maxKeep: int) -> List[float]:
        if key not in traderObject:
            traderObject[key] = []
        history = traderObject[key]
        history.append(round(value, 4))
        if len(history) > maxKeep:
            history[:] = history[-maxKeep:]
        return history

    def average(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def stddev(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        meanValue = self.average(values)
        variance = sum((value - meanValue) ** 2 for value in values) / len(values)
        return math.sqrt(max(variance, 0.0))

    def clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def maxBuyCapacity(self, product: str, position: int, buyOrderVolume: int) -> int:
        return self.POSITION_LIMITS[product] - position - buyOrderVolume

    def maxSellCapacity(self, product: str, position: int, sellOrderVolume: int) -> int:
        return self.POSITION_LIMITS[product] + position - sellOrderVolume

    def linearRegressionFairValue(self, values: List[float], window: int) -> float:
        n = min(len(values), window)
        if n == 0:
            return 0.0
        if n == 1:
            return values[-1]

        y = values[-n:]
        meanX = (n - 1) / 2.0
        meanY = sum(y) / n

        numerator = 0.0
        denominator = 0.0
        for i, value in enumerate(y):
            dx = i - meanX
            numerator += dx * (value - meanY)
            denominator += dx * dx

        slope = 0.0 if denominator == 0 else numerator / denominator
        intercept = meanY - slope * meanX

        # Current fitted level, not one-step-ahead trend extrapolation.
        return slope * (n - 1) + intercept

    def getVolumeWeightedLevelPrice(self, levels: List[Tuple[int, int]], isBid: bool) -> Optional[int]:
        if not levels:
            return None

        bestPrice = levels[0][0]
        bestAbsVolume = abs(levels[0][1])

        for price, volume in levels[1:]:
            absVolume = abs(volume)
            if absVolume > bestAbsVolume:
                bestPrice = price
                bestAbsVolume = absVolume
            elif absVolume == bestAbsVolume:
                if isBid and price > bestPrice:
                    bestPrice = price
                if not isBid and price < bestPrice:
                    bestPrice = price

        return bestPrice

    def orderBookImbalance(self, orderDepth: OrderDepth) -> float:
        bidVolume = sum(max(volume, 0) for _, volume in self.sortedBids(orderDepth))
        askVolume = sum(abs(volume) for _, volume in self.sortedAsks(orderDepth))
        total = bidVolume + askVolume
        if total == 0:
            return 0.0
        return (bidVolume - askVolume) / total

    def updatePepperFairValue(
        self,
        traderObject: dict,
        product: str,
        orderDepth: OrderDepth,
    ) -> Optional[Dict[str, float]]:
        params = PARAMS[product]
        midPrice = self.getMidPrice(orderDepth)
        if midPrice is None:
            return None

        midHistory = self.appendHistory(
            traderObject,
            f"{product}_mid_history",
            midPrice,
            int(params["history_keep"]),
        )

        fairValue = self.linearRegressionFairValue(midHistory, int(params["regression_window"]))
        residual = midPrice - fairValue

        residualHistory = self.appendHistory(
            traderObject,
            f"{product}_residual_history",
            residual,
            int(params["residual_window"]),
        )

        residualStd = max(
            params["min_residual_std"],
            self.stddev(residualHistory[-int(params["residual_window"]):]),
        )
        residualZ = residual / residualStd if residualStd > 0 else 0.0

        return {
            "mid": midPrice,
            "fair": fairValue,
            "residual": residual,
            "residualStd": residualStd,
            "residualZ": residualZ,
        }

    def updateOsmiumFairValue(
        self,
        traderObject: dict,
        product: str,
        orderDepth: OrderDepth,
    ) -> Optional[Dict[str, float]]:
        params = PARAMS[product]

        bestBid = self.getBestBid(orderDepth)
        bestAsk = self.getBestAsk(orderDepth)
        if bestBid is None or bestAsk is None:
            return None

        bidLevels = self.sortedBids(orderDepth)
        askLevels = self.sortedAsks(orderDepth)

        wallBid = self.getVolumeWeightedLevelPrice(bidLevels, True)
        wallAsk = self.getVolumeWeightedLevelPrice(askLevels, False)

        if wallBid is None:
            wallBid = bestBid
        if wallAsk is None:
            wallAsk = bestAsk

        wallMid = (wallBid + wallAsk) / 2.0
        wallHistory = self.appendHistory(
            traderObject,
            f"{product}_wall_mid_history",
            wallMid,
            int(params["history_keep"]),
        )

        meanWindow = int(params["mean_window"])
        rollingMean = self.average(wallHistory[-meanWindow:])

        lastReturn = 0.0
        if len(wallHistory) >= 2:
            lastReturn = wallHistory[-1] - wallHistory[-2]

        momentum5 = 0.0
        if len(wallHistory) >= 6:
            momentum5 = wallHistory[-1] - wallHistory[-6]

        imbalance = self.orderBookImbalance(orderDepth)

        fairValue = (
            rollingMean
            - params["last_return_coeff"] * lastReturn
            - params["momentum5_coeff"] * momentum5
            + params["imbalance_coeff"] * imbalance
        )

        return {
            "mid": (bestBid + bestAsk) / 2.0,
            "fair": fairValue,
            "wallMid": wallMid,
            "rollingMean": rollingMean,
            "lastReturn": lastReturn,
            "momentum5": momentum5,
            "imbalance": imbalance,
        }

    def maybeJoinBid(self, fairBid: int, bestBid: Optional[int], bestAsk: Optional[int], joinThreshold: float) -> int:
        if bestBid is None:
            return fairBid
        joinedBid = fairBid
        if bestAsk is not None and bestBid + 1 < bestAsk and fairBid - bestBid >= joinThreshold:
            joinedBid = max(joinedBid, bestBid + 1)
        return joinedBid

    def maybeJoinAsk(self, fairAsk: int, bestBid: Optional[int], bestAsk: Optional[int], joinThreshold: float) -> int:
        if bestAsk is None:
            return fairAsk
        joinedAsk = fairAsk
        if bestBid is not None and bestAsk - 1 > bestBid and bestAsk - fairAsk >= joinThreshold:
            joinedAsk = min(joinedAsk, bestAsk - 1)
        return joinedAsk


    def formatLevels(self, levels: List[Tuple[int, int]], maxLevels: int = 3) -> List[Tuple[int, int]]:
        return [(int(price), int(volume)) for price, volume in levels[:maxLevels]]

    def summarizeOwnTrades(self, state: TradingState, product: str, maxTrades: int = 3) -> List[Tuple[int, int, int]]:
        trades = state.own_trades.get(product, []) if hasattr(state, "own_trades") else []
        recent = trades[-maxTrades:]
        return [(int(trade.price), int(trade.quantity), int(trade.timestamp)) for trade in recent]

    def tradePepper(
        self,
        product: str,
        orderDepth: OrderDepth,
        signal: Optional[Dict[str, float]],
        position: int,
        timestamp: int,
    ) -> List[Order]:
        if signal is None:
            return []

        params = PARAMS[product]
        fairValue = signal["fair"]
        residualZ = signal["residualZ"]
        bestBid = self.getBestBid(orderDepth)
        bestAsk = self.getBestAsk(orderDepth)

        orders: List[Order] = []
        buyOrderVolume = 0
        sellOrderVolume = 0
        takenBuys: List[Tuple[int, int]] = []
        takenSells: List[Tuple[int, int]] = []

        if timestamp < params["early_end_ts"]:
            targetPos = int(params["target_early"])
            holdPremium = params["hold_premium_early"]
        elif timestamp < params["mid_end_ts"]:
            targetPos = int(params["target_mid"])
            holdPremium = params["hold_premium_mid"]
        else:
            targetPos = int(params["target_late"])
            holdPremium = params["hold_premium_late"]

        # Pepper thesis: get long early, avoid selling that inventory cheaply.
        # Residual z only nudges entry aggressiveness a bit.
        if residualZ <= -1.0:
            holdPremium += 2.0
        elif residualZ >= 1.25:
            holdPremium -= 2.0

        # inventory-dependent fair used mainly for quote placement
        inventorySkew = max(0.0, position - targetPos) / 10.0 * params["inventory_skew_per_10"]
        adjustedFair = fairValue + holdPremium - inventorySkew

        # Aggressively accumulate toward target on asks up to the biased fair.
        buyCapToTarget = max(0, targetPos - position)
        buyThreshold = adjustedFair
        for askPrice, askVolume in self.sortedAsks(orderDepth):
            if buyCapToTarget <= buyOrderVolume:
                break
            available = -askVolume
            if askPrice <= buyThreshold:
                qty = min(
                    available,
                    buyCapToTarget - buyOrderVolume,
                    self.maxBuyCapacity(product, position, buyOrderVolume),
                )
                if qty > 0:
                    orders.append(Order(product, askPrice, qty))
                    buyOrderVolume += qty
                    takenBuys.append((askPrice, qty))
            else:
                break

        positionAfterTake = position + buyOrderVolume - sellOrderVolume

        # Only sell Pepper inventory if it is very rich, or very late and we are over target.
        richSellEdge = params["rich_sell_edge"]
        if positionAfterTake > targetPos + 6 or timestamp >= params["late_reduce_ts"]:
            sellThreshold = fairValue + max(2.0, richSellEdge - holdPremium)
            if timestamp >= params["late_reduce_ts"]:
                sellThreshold = fairValue + max(0.5, params["clear_edge"])
            for bidPrice, bidVolume in self.sortedBids(orderDepth):
                if bidPrice >= sellThreshold:
                    desired = max(0, positionAfterTake - targetPos)
                    if timestamp >= params["late_reduce_ts"]:
                        desired = max(desired, max(0, positionAfterTake - max(40, targetPos - 20)))
                    if desired <= 0:
                        break
                    qty = min(
                        bidVolume,
                        desired,
                        self.maxSellCapacity(product, position, sellOrderVolume),
                    )
                    if qty > 0:
                        orders.append(Order(product, bidPrice, -qty))
                        sellOrderVolume += qty
                        positionAfterTake -= qty
                        takenSells.append((bidPrice, qty))
                else:
                    break

        # Passive quote logic: mostly bid while under target, tiny asks only when over target or late.
        bidQuote = None
        buyQty = 0
        if positionAfterTake < targetPos:
            fairBid = math.floor(adjustedFair - 1)
            bidQuote = self.maybeJoinBid(fairBid, bestBid, bestAsk, max(2.0, params["join_threshold"] - 2.0))
            bidSize = int(params["base_order_size"]) + 6
            gapToTarget = targetPos - positionAfterTake
            if gapToTarget >= 40:
                bidSize += 8
            elif gapToTarget >= 20:
                bidSize += 4
            buyQty = min(bidSize, self.maxBuyCapacity(product, position, buyOrderVolume))
            if buyQty > 0:
                orders.append(Order(product, int(bidQuote), int(buyQty)))

        # Ask only sparingly so we do not constantly scalp away long inventory.
        askQty = 0
        askQuote = None
        if positionAfterTake > targetPos + 10 or timestamp >= params["late_reduce_ts"]:
            askBase = fairValue + max(4.0, holdPremium + 2.0)
            if timestamp >= params["late_reduce_ts"]:
                askBase = fairValue + 1.0
            askQuote = self.maybeJoinAsk(math.ceil(askBase), bestBid, bestAsk, params["join_threshold"])
            askSize = int(params["base_order_size"]) // 2
            if positionAfterTake > targetPos + 20:
                askSize += 4
            askQty = min(max(0, askSize), self.maxSellCapacity(product, position, sellOrderVolume))
            if askQty > 0:
                orders.append(Order(product, int(askQuote), -int(askQty)))

        if self.shouldLogProduct(product):
            logger.print(
                product,
                "signal",
                "mid =", round(signal["mid"], 2),
                "fair =", round(fairValue, 2),
                "adjFair =", round(adjustedFair, 2),
                "resid =", round(signal["residual"], 2),
                "residStd =", round(signal["residualStd"], 2),
                "residZ =", round(residualZ, 2),
                "targetPos =", targetPos,
                "holdPremium =", round(holdPremium, 2),
                "pos =", position,
                "afterTake =", positionAfterTake,
            )
            quotePreview = []
            if bidQuote is not None and buyQty > 0:
                quotePreview.append((int(bidQuote), int(buyQty)))
            if askQuote is not None and askQty > 0:
                quotePreview.append((int(askQuote), -int(askQty)))
            logger.print(
                product,
                "actions",
                "takenBuys =", takenBuys[:4],
                "takenSells =", takenSells[:4],
                "quotes =", quotePreview,
                "finalOrders =", [(order.price, order.quantity) for order in orders[:8]],
            )

        return orders


    def tradeOsmium(
        self,
        product: str,
        orderDepth: OrderDepth,
        signal: Optional[Dict[str, float]],
        position: int,
    ) -> List[Order]:
        if signal is None:
            return []

        params = PARAMS[product]
        fairValue = signal["fair"]
        imbalance = signal["imbalance"]

        bestBid = self.getBestBid(orderDepth)
        bestAsk = self.getBestAsk(orderDepth)

        if bestBid is None or bestAsk is None:
            return []

        spread = bestAsk - bestBid
        orders: List[Order] = []
        buyOrderVolume = 0
        sellOrderVolume = 0
        takenBuys: List[Tuple[int, int]] = []
        takenSells: List[Tuple[int, int]] = []

        inventorySkew = (position / 8.0) * params["inventory_skew_per_8"]
        adjustedFair = fairValue - inventorySkew
        midPrice = signal["mid"]
        directionalGap = adjustedFair - midPrice

        buyTakeEdge = params["take_edge"]
        sellTakeEdge = params["take_edge"]
        buyTakeBonus = 0
        sellTakeBonus = 0

        if imbalance >= 0.22:
            buyTakeEdge -= 0.75
        elif imbalance <= -0.22:
            sellTakeEdge -= 0.75

        if directionalGap >= 1.0:
            buyTakeEdge -= 0.50
        elif directionalGap <= -1.0:
            sellTakeEdge -= 0.50

        # strongest taker mode when the fair-gap and imbalance agree
        if directionalGap >= 1.40 and imbalance >= 0.12:
            buyTakeEdge -= 0.75
            buyTakeBonus += 6
        elif directionalGap <= -1.40 and imbalance <= -0.12:
            sellTakeEdge -= 0.75
            sellTakeBonus += 6

        if directionalGap >= 2.20:
            buyTakeEdge -= 0.50
            buyTakeBonus += 4
        elif directionalGap <= -2.20:
            sellTakeEdge -= 0.50
            sellTakeBonus += 4

        buyTakeEdge = max(1.0, buyTakeEdge)
        sellTakeEdge = max(1.0, sellTakeEdge)

        maxTakePerSide = int(params["base_order_size"]) + 10

        for askPrice, askVolume in self.sortedAsks(orderDepth):
            available = -askVolume
            if askPrice <= adjustedFair - buyTakeEdge:
                remaining = min(
                    maxTakePerSide + buyTakeBonus - buyOrderVolume,
                    self.maxBuyCapacity(product, position, buyOrderVolume),
                )
                qty = min(available, max(0, remaining))
                if qty > 0:
                    orders.append(Order(product, askPrice, qty))
                    buyOrderVolume += qty
                    takenBuys.append((askPrice, qty))
            else:
                break

        for bidPrice, bidVolume in self.sortedBids(orderDepth):
            available = bidVolume
            if bidPrice >= adjustedFair + sellTakeEdge:
                remaining = min(
                    maxTakePerSide + sellTakeBonus - sellOrderVolume,
                    self.maxSellCapacity(product, position, sellOrderVolume),
                )
                qty = min(available, max(0, remaining))
                if qty > 0:
                    orders.append(Order(product, bidPrice, -qty))
                    sellOrderVolume += qty
                    takenSells.append((bidPrice, qty))
            else:
                break

        positionAfterTake = position + buyOrderVolume - sellOrderVolume

        # clear inventory with active taking near fair
        if positionAfterTake > 28:
            for bidPrice, bidVolume in self.sortedBids(orderDepth):
                if bidPrice >= adjustedFair - params["clear_edge"]:
                    qty = min(
                        bidVolume,
                        positionAfterTake - 28,
                        self.maxSellCapacity(product, position, sellOrderVolume),
                    )
                    if qty > 0:
                        orders.append(Order(product, bidPrice, -qty))
                        sellOrderVolume += qty
                        positionAfterTake -= qty
                        takenSells.append((bidPrice, qty))
                else:
                    break
        elif positionAfterTake < -28:
            for askPrice, askVolume in self.sortedAsks(orderDepth):
                if askPrice <= adjustedFair + params["clear_edge"]:
                    qty = min(
                        -askVolume,
                        (-28) - positionAfterTake,
                        self.maxBuyCapacity(product, position, buyOrderVolume),
                    )
                    if qty > 0:
                        orders.append(Order(product, askPrice, qty))
                        buyOrderVolume += qty
                        positionAfterTake += qty
                        takenBuys.append((askPrice, qty))
                else:
                    break

        # one-sided / signal-gated market making
        signalAbs = abs(directionalGap)
        imbalanceAbs = abs(imbalance)
        wideSpread = spread >= 12
        veryWideSpread = spread >= 16

        dominantBuy = directionalGap >= 0.90 or (directionalGap >= 0.35 and imbalance >= 0.20)
        dominantSell = directionalGap <= -0.90 or (directionalGap <= -0.35 and imbalance <= -0.20)
        weakSignal = signalAbs < 0.80 and imbalanceAbs < 0.15

        bidQuote = None
        askQuote = None
        buySize = 0
        sellSize = 0
        quoteMode = "off"

        baseSize = int(params["base_order_size"])

        if dominantBuy and wideSpread:
            fairBid = math.floor(adjustedFair - max(1.0, params["quote_edge"] - 1.5))
            bidQuote = self.maybeJoinBid(fairBid, bestBid, bestAsk, max(2.0, params["join_threshold"] - 3.0))
            buySize = baseSize + 2
            if directionalGap >= 1.8:
                buySize += 4
            if imbalance >= 0.30:
                buySize += 2
            quoteMode = "buy-lean"

            if positionAfterTake > 10:
                askQuote = self.maybeJoinAsk(math.ceil(adjustedFair + params["quote_edge"]), bestBid, bestAsk, params["join_threshold"])
                sellSize = max(2, baseSize // 3 + 2)
                quoteMode = "buy-lean+trim"

        elif dominantSell and wideSpread:
            fairAsk = math.ceil(adjustedFair + max(1.0, params["quote_edge"] - 1.5))
            askQuote = self.maybeJoinAsk(fairAsk, bestBid, bestAsk, max(2.0, params["join_threshold"] - 3.0))
            sellSize = baseSize + 2
            if directionalGap <= -1.8:
                sellSize += 4
            if imbalance <= -0.30:
                sellSize += 2
            quoteMode = "sell-lean"

            if positionAfterTake < -10:
                bidQuote = self.maybeJoinBid(math.floor(adjustedFair - params["quote_edge"]), bestBid, bestAsk, params["join_threshold"])
                buySize = max(2, baseSize // 3 + 2)
                quoteMode = "sell-lean+trim"

        elif veryWideSpread and not weakSignal:
            # moderate signal: quote both sides, but sizes stay small and inventory-aware
            fairBid = math.floor(adjustedFair - params["quote_edge"])
            fairAsk = math.ceil(adjustedFair + params["quote_edge"])
            bidQuote = self.maybeJoinBid(fairBid, bestBid, bestAsk, params["join_threshold"])
            askQuote = self.maybeJoinAsk(fairAsk, bestBid, bestAsk, params["join_threshold"])
            buySize = max(3, baseSize // 2)
            sellSize = max(3, baseSize // 2)
            quoteMode = "tiny-both"
        elif positionAfterTake > 8 and wideSpread:
            askQuote = self.maybeJoinAsk(math.ceil(adjustedFair + max(1.0, params["quote_edge"] - 1.0)), bestBid, bestAsk, max(2.0, params["join_threshold"] - 2.0))
            sellSize = max(2, baseSize // 3 + 2)
            quoteMode = "trim-long"
        elif positionAfterTake < -8 and wideSpread:
            bidQuote = self.maybeJoinBid(math.floor(adjustedFair - max(1.0, params["quote_edge"] - 1.0)), bestBid, bestAsk, max(2.0, params["join_threshold"] - 2.0))
            buySize = max(2, baseSize // 3 + 2)
            quoteMode = "trim-short"

        # inventory overrides: suppress same-side quotes earlier
        if positionAfterTake > 12:
            if askQuote is not None:
                askQuote -= 1
            sellSize += 3
            if directionalGap < 2.6:
                buySize = 0
            else:
                buySize = max(0, buySize - 5)
        elif positionAfterTake < -12:
            if bidQuote is not None:
                bidQuote += 1
            buySize += 3
            if directionalGap > -2.6:
                sellSize = 0
            else:
                sellSize = max(0, sellSize - 5)

        if positionAfterTake > params["soft_position_limit"]:
            if askQuote is not None:
                askQuote -= 1
            sellSize += 5
            buySize = 0
            quoteMode += "|hard-trim-long"
        elif positionAfterTake < -params["soft_position_limit"]:
            if bidQuote is not None:
                bidQuote += 1
            buySize += 5
            sellSize = 0
            quoteMode += "|hard-trim-short"

        if bidQuote is not None and askQuote is not None and bidQuote >= askQuote:
            askQuote = None if sellSize <= buySize else math.ceil(adjustedFair + 1)
            bidQuote = None if buySize <= sellSize else math.floor(adjustedFair - 1)

        buyQty = 0
        sellQty = 0

        if bidQuote is not None and buySize > 0:
            buyQty = min(max(0, buySize), self.maxBuyCapacity(product, position, buyOrderVolume))
            if buyQty > 0:
                orders.append(Order(product, int(bidQuote), int(buyQty)))

        if askQuote is not None and sellSize > 0:
            sellQty = min(max(0, sellSize), self.maxSellCapacity(product, position, sellOrderVolume))
            if sellQty > 0:
                orders.append(Order(product, int(askQuote), -int(sellQty)))

        if self.shouldLogProduct(product):
            logger.print(
                product,
                "signal",
                "mid =", round(signal["mid"], 2),
                "wallMid =", round(signal["wallMid"], 2),
                "mean20 =", round(signal["rollingMean"], 2),
                "fair =", round(fairValue, 2),
                "adjFair =", round(adjustedFair, 2),
                "gap =", round(directionalGap, 2),
                "lastRet =", round(signal["lastReturn"], 2),
                "mom5 =", round(signal["momentum5"], 2),
                "imb =", round(imbalance, 3),
                "spread =", spread,
                "takeEdges =", (round(buyTakeEdge, 2), round(sellTakeEdge, 2)),
                "mode =", quoteMode,
                "pos =", position,
                "afterTake =", positionAfterTake,
            )
            quotePreview = []
            if bidQuote is not None and buyQty > 0:
                quotePreview.append((int(bidQuote), int(buyQty)))
            if askQuote is not None and sellQty > 0:
                quotePreview.append((int(askQuote), -int(sellQty)))
            logger.print(
                product,
                "actions",
                "takenBuys =", takenBuys[:4],
                "takenSells =", takenSells[:4],
                "quotes =", quotePreview,
                "finalOrders =", [(order.price, order.quantity) for order in orders[:8]],
            )

        return orders
