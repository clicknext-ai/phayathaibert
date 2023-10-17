from datasets import (
    GeneratorBasedBuilder,
    DatasetInfo,
    Features,
    Value,
    Sequence,
    ClassLabel,
    SplitGenerator,
    Split
)
from os import path
from glob import glob

class ThaiNNER(GeneratorBasedBuilder):

    _SENTENCE_SPLITTER = '\n'
    _TRAIN_FOLDER = "train"
    _TEST_FOLDER = "test"
    _LAYER_1_TAGS =  ['B-address', 'B-airport', 'B-animal_species', 'B-animate', 'B-army', 'B-award', 'B-band', 'B-book', 'B-bridge', 'B-building', 'B-cardinal', 'B-city', 'B-concert', 'B-continent', 'B-country', 'B-date', 'B-day', 'B-disease', 'B-distance', 'B-district', 'B-duration', 'B-electronics', 'B-energy', 'B-event:others', 'B-facility:other', 'B-film', 'B-firstname', 'B-fold', 'B-food:ingredient', 'B-fund', 'B-game', 'B-god', 'B-goverment', 'B-hospital', 'B-hotel', 'B-index', 'B-island', 'B-jargon', 'B-language', 'B-last', 'B-latitude', 'B-law', 'B-loc:others', 'B-longtitude', 'B-media', 'B-money', 'B-month', 'B-mountian', 'B-mult', 'B-museum', 'B-namemod', 'B-nationality', 'B-natural_disaster', 'B-nickname', 'B-nicknametitle', 'B-norp:others', 'B-norp:political', 'B-ocean', 'B-org:edu', 'B-org:other', 'B-org:political', 'B-org:religious', 'B-orgcorp', 'B-percent', 'B-periodic', 'B-person', 'B-port', 'B-postcode', 'B-product:drug', 'B-product:food', 'B-province', 'B-psudoname', 'B-quantity', 'B-rel', 'B-religion', 'B-restaurant', 'B-river', 'B-roadname', 'B-role', 'B-sciname', 'B-season', 'B-soi', 'B-song', 'B-space', 'B-speed', 'B-sports_event', 'B-sports_team', 'B-stadium', 'B-state', 'B-station', 'B-stock_exchange', 'B-sub_district', 'B-temperature', 'B-time', 'B-title', 'B-tv_show', 'B-unit', 'B-vehicle', 'B-war', 'B-weapon', 'B-weight', 'B-woa', 'B-year', 'E-address', 'E-airport', 'E-animal_species', 'E-animate', 'E-army', 'E-award', 'E-band', 'E-book', 'E-bridge', 'E-building', 'E-cardinal', 'E-city', 'E-concert', 'E-continent', 'E-country', 'E-date', 'E-day', 'E-disease', 'E-distance', 'E-district', 'E-duration', 'E-electronics', 'E-energy', 'E-event:others', 'E-facility:other', 'E-film', 'E-firstname', 'E-fold', 'E-food:ingredient', 'E-fund', 'E-game', 'E-god', 'E-goverment', 'E-hospital', 'E-hotel', 'E-index', 'E-island', 'E-jargon', 'E-language', 'E-last', 'E-latitude', 'E-law', 'E-loc:others', 'E-longtitude', 'E-media', 'E-money', 'E-month', 'E-mountian', 'E-mult', 'E-museum', 'E-namemod', 'E-nationality', 'E-natural_disaster', 'E-nickname', 'E-nicknametitle', 'E-norp:others', 'E-norp:political', 'E-ocean', 'E-org:edu', 'E-org:other', 'E-org:political', 'E-org:religious', 'E-orgcorp', 'E-percent', 'E-periodic', 'E-person', 'E-port', 'E-postcode', 'E-product:drug', 'E-product:food', 'E-province', 'E-psudoname', 'E-quantity', 'E-rel', 'E-religion', 'E-restaurant', 'E-river', 'E-roadname', 'E-role', 'E-sciname', 'E-season', 'E-soi', 'E-song', 'E-space', 'E-speed', 'E-sports_event', 'E-sports_team', 'E-stadium', 'E-state', 'E-station', 'E-stock_exchange', 'E-sub_district', 'E-temperature', 'E-time', 'E-title', 'E-tv_show', 'E-unit', 'E-vehicle', 'E-war', 'E-weapon', 'E-weight', 'E-woa', 'E-year', 'I-address', 'I-airport', 'I-animal_species', 'I-animate', 'I-army', 'I-award', 'I-band', 'I-book', 'I-bridge', 'I-building', 'I-cardinal', 'I-city', 'I-concert', 'I-continent', 'I-country', 'I-date', 'I-day', 'I-disease', 'I-distance', 'I-district', 'I-duration', 'I-electronics', 'I-energy', 'I-event:others', 'I-facility:other', 'I-film', 'I-firstname', 'I-fold', 'I-food:ingredient', 'I-fund', 'I-game', 'I-god', 'I-goverment', 'I-hospital', 'I-hotel', 'I-index', 'I-island', 'I-jargon', 'I-language', 'I-last', 'I-latitude', 'I-law', 'I-loc:others', 'I-longtitude', 'I-media', 'I-money', 'I-month', 'I-mountian', 'I-museum', 'I-namemod', 'I-nationality', 'I-natural_disaster', 'I-nickname', 'I-norp:others', 'I-norp:political', 'I-org:edu', 'I-org:other', 'I-org:political', 'I-org:religious', 'I-orgcorp', 'I-percent', 'I-periodic', 'I-person', 'I-port', 'I-product:drug', 'I-product:food', 'I-province', 'I-psudoname', 'I-quantity', 'I-rel', 'I-religion', 'I-restaurant', 'I-river', 'I-roadname', 'I-role', 'I-sciname', 'I-season', 'I-soi', 'I-song', 'I-space', 'I-speed', 'I-sports_event', 'I-sports_team', 'I-stadium', 'I-state', 'I-station', 'I-stock_exchange', 'I-sub_district', 'I-temperature', 'I-time', 'I-title', 'I-tv_show', 'I-unit', 'I-vehicle', 'I-war', 'I-weapon', 'I-weight', 'I-woa', 'I-year', 'O', 'S-address', 'S-airport', 'S-animal_species', 'S-animate', 'S-army', 'S-award', 'S-band', 'S-book', 'S-bridge', 'S-building', 'S-cardinal', 'S-city', 'S-continent', 'S-country', 'S-date', 'S-day', 'S-disease', 'S-district', 'S-duration', 'S-electronics', 'S-event:others', 'S-facility:other', 'S-film', 'S-firstname', 'S-fold', 'S-food:ingredient', 'S-fund', 'S-game', 'S-god', 'S-goverment', 'S-hospital', 'S-hotel', 'S-island', 'S-jargon', 'S-language', 'S-last', 'S-law', 'S-loc:others', 'S-media', 'S-money', 'S-month', 'S-mountian', 'S-mult', 'S-museum', 'S-namemod', 'S-nationality', 'S-natural_disaster', 'S-nickname', 'S-nicknametitle', 'S-norp:others', 'S-norp:political', 'S-ocean', 'S-org:edu', 'S-org:other', 'S-org:political', 'S-org:religious', 'S-orgcorp', 'S-percent', 'S-periodic', 'S-person', 'S-product:drug', 'S-product:food', 'S-province', 'S-psudoname', 'S-quantity', 'S-rel', 'S-religion', 'S-restaurant', 'S-river', 'S-roadname', 'S-role', 'S-sciname', 'S-season', 'S-soi', 'S-song', 'S-space', 'S-sports_event', 'S-sports_team', 'S-stadium', 'S-state', 'S-station', 'S-stock_exchange', 'S-sub_district', 'S-time', 'S-title', 'S-tv_show', 'S-unit', 'S-vehicle', 'S-war', 'S-weapon', 'S-weight', 'S-woa', 'S-year']
    _LAYER_2_TAGS =  ['B-address', 'B-airport', 'B-animal_species', 'B-animate', 'B-army', 'B-award', 'B-band', 'B-book', 'B-bridge', 'B-building', 'B-cardinal', 'B-city', 'B-continent', 'B-country', 'B-date', 'B-day', 'B-disease', 'B-distance', 'B-district', 'B-duration', 'B-electronics', 'B-event:others', 'B-facility:other', 'B-film', 'B-firstname', 'B-food:ingredient', 'B-fund', 'B-game', 'B-god', 'B-goverment', 'B-hospital', 'B-hotel', 'B-island', 'B-jargon', 'B-language', 'B-last', 'B-latitude', 'B-law', 'B-loc:others', 'B-longtitude', 'B-media', 'B-middle', 'B-money', 'B-month', 'B-mountian', 'B-mult', 'B-museum', 'B-namemod', 'B-nationality', 'B-natural_disaster', 'B-nickname', 'B-nicknametitle', 'B-norp:others', 'B-norp:political', 'B-ocean', 'B-org:edu', 'B-org:other', 'B-org:political', 'B-org:religious', 'B-orgcorp', 'B-percent', 'B-periodic', 'B-person', 'B-port', 'B-product:drug', 'B-product:food', 'B-province', 'B-psudoname', 'B-quantity', 'B-rel', 'B-religion', 'B-restaurant', 'B-river', 'B-roadname', 'B-role', 'B-sciname', 'B-season', 'B-soi', 'B-song', 'B-space', 'B-sports_event', 'B-sports_team', 'B-stadium', 'B-state', 'B-station', 'B-stock_exchange', 'B-sub_district', 'B-time', 'B-title', 'B-tv_show', 'B-unit', 'B-vehicle', 'B-war', 'B-weapon', 'B-woa', 'B-year', 'E-address', 'E-airport', 'E-animal_species', 'E-animate', 'E-army', 'E-award', 'E-band', 'E-book', 'E-bridge', 'E-building', 'E-cardinal', 'E-city', 'E-continent', 'E-country', 'E-date', 'E-day', 'E-disease', 'E-distance', 'E-district', 'E-duration', 'E-electronics', 'E-event:others', 'E-facility:other', 'E-film', 'E-firstname', 'E-food:ingredient', 'E-fund', 'E-game', 'E-god', 'E-goverment', 'E-hospital', 'E-hotel', 'E-island', 'E-jargon', 'E-language', 'E-last', 'E-latitude', 'E-law', 'E-loc:others', 'E-longtitude', 'E-media', 'E-middle', 'E-money', 'E-month', 'E-mountian', 'E-mult', 'E-museum', 'E-namemod', 'E-nationality', 'E-natural_disaster', 'E-nickname', 'E-nicknametitle', 'E-norp:others', 'E-norp:political', 'E-ocean', 'E-org:edu', 'E-org:other', 'E-org:political', 'E-org:religious', 'E-orgcorp', 'E-percent', 'E-periodic', 'E-person', 'E-port', 'E-product:drug', 'E-product:food', 'E-province', 'E-psudoname', 'E-quantity', 'E-rel', 'E-religion', 'E-restaurant', 'E-river', 'E-roadname', 'E-role', 'E-sciname', 'E-season', 'E-soi', 'E-song', 'E-space', 'E-sports_event', 'E-sports_team', 'E-stadium', 'E-state', 'E-station', 'E-stock_exchange', 'E-sub_district', 'E-time', 'E-title', 'E-tv_show', 'E-unit', 'E-vehicle', 'E-war', 'E-weapon', 'E-woa', 'E-year', 'I-address', 'I-airport', 'I-animal_species', 'I-animate', 'I-army', 'I-award', 'I-band', 'I-book', 'I-bridge', 'I-building', 'I-cardinal', 'I-city', 'I-continent', 'I-country', 'I-date', 'I-day', 'I-disease', 'I-distance', 'I-district', 'I-duration', 'I-electronics', 'I-event:others', 'I-facility:other', 'I-film', 'I-firstname', 'I-food:ingredient', 'I-fund', 'I-game', 'I-god', 'I-goverment', 'I-hospital', 'I-hotel', 'I-island', 'I-jargon', 'I-language', 'I-last', 'I-latitude', 'I-law', 'I-loc:others', 'I-longtitude', 'I-media', 'I-middle', 'I-money', 'I-month', 'I-mountian', 'I-museum', 'I-namemod', 'I-nationality', 'I-natural_disaster', 'I-nickname', 'I-norp:others', 'I-norp:political', 'I-org:edu', 'I-org:other', 'I-org:political', 'I-org:religious', 'I-orgcorp', 'I-percent', 'I-periodic', 'I-person', 'I-port', 'I-product:drug', 'I-product:food', 'I-province', 'I-psudoname', 'I-quantity', 'I-rel', 'I-religion', 'I-restaurant', 'I-river', 'I-roadname', 'I-role', 'I-sciname', 'I-season', 'I-soi', 'I-song', 'I-space', 'I-sports_event', 'I-sports_team', 'I-stadium', 'I-state', 'I-station', 'I-stock_exchange', 'I-sub_district', 'I-time', 'I-title', 'I-tv_show', 'I-unit', 'I-vehicle', 'I-war', 'I-weapon', 'I-woa', 'I-year', 'O', 'S-address', 'S-airport', 'S-animal_species', 'S-animate', 'S-army', 'S-award', 'S-band', 'S-book', 'S-bridge', 'S-building', 'S-cardinal', 'S-city', 'S-continent', 'S-country', 'S-date', 'S-day', 'S-disease', 'S-distance', 'S-district', 'S-duration', 'S-electronics', 'S-event:others', 'S-facility:other', 'S-film', 'S-firstname', 'S-fold', 'S-food:ingredient', 'S-fund', 'S-game', 'S-god', 'S-goverment', 'S-hospital', 'S-hotel', 'S-island', 'S-jargon', 'S-language', 'S-last', 'S-law', 'S-loc:others', 'S-longtitude', 'S-media', 'S-middle', 'S-money', 'S-month', 'S-mountian', 'S-mult', 'S-museum', 'S-namemod', 'S-nationality', 'S-natural_disaster', 'S-nickname', 'S-nicknametitle', 'S-norp:others', 'S-norp:political', 'S-ocean', 'S-org:edu', 'S-org:other', 'S-org:political', 'S-org:religious', 'S-orgcorp', 'S-percent', 'S-periodic', 'S-person', 'S-port', 'S-postcode', 'S-product:drug', 'S-product:food', 'S-province', 'S-psudoname', 'S-quantity', 'S-rel', 'S-religion', 'S-restaurant', 'S-river', 'S-roadname', 'S-role', 'S-sciname', 'S-season', 'S-soi', 'S-song', 'S-space', 'S-sports_event', 'S-sports_team', 'S-stadium', 'S-state', 'S-station', 'S-stock_exchange', 'S-sub_district', 'S-time', 'S-title', 'S-tv_show', 'S-unit', 'S-vehicle', 'S-war', 'S-weapon', 'S-weight', 'S-woa', 'S-year']
    _LAYER_3_TAGS =  ['B-address', 'B-animal_species', 'B-animate', 'B-army', 'B-book', 'B-bridge', 'B-building', 'B-cardinal', 'B-city', 'B-continent', 'B-country', 'B-date', 'B-day', 'B-disease', 'B-district', 'B-duration', 'B-electronics', 'B-event:others', 'B-facility:other', 'B-firstname', 'B-food:ingredient', 'B-fund', 'B-goverment', 'B-hospital', 'B-hotel', 'B-island', 'B-jargon', 'B-last', 'B-law', 'B-loc:others', 'B-media', 'B-middle', 'B-month', 'B-mountian', 'B-mult', 'B-namemod', 'B-nationality', 'B-nickname', 'B-nicknametitle', 'B-norp:others', 'B-norp:political', 'B-ocean', 'B-org:edu', 'B-org:other', 'B-org:political', 'B-org:religious', 'B-orgcorp', 'B-periodic', 'B-person', 'B-port', 'B-product:drug', 'B-product:food', 'B-province', 'B-psudoname', 'B-quantity', 'B-rel', 'B-religion', 'B-restaurant', 'B-river', 'B-roadname', 'B-role', 'B-sciname', 'B-soi', 'B-song', 'B-space', 'B-sports_event', 'B-state', 'B-station', 'B-sub_district', 'B-time', 'B-title', 'B-tv_show', 'B-unit', 'B-vehicle', 'B-weapon', 'B-woa', 'B-year', 'E-address', 'E-animal_species', 'E-animate', 'E-army', 'E-book', 'E-bridge', 'E-building', 'E-cardinal', 'E-city', 'E-continent', 'E-country', 'E-date', 'E-day', 'E-disease', 'E-district', 'E-duration', 'E-electronics', 'E-event:others', 'E-facility:other', 'E-firstname', 'E-food:ingredient', 'E-fund', 'E-goverment', 'E-hospital', 'E-hotel', 'E-island', 'E-jargon', 'E-last', 'E-law', 'E-loc:others', 'E-media', 'E-middle', 'E-month', 'E-mountian', 'E-mult', 'E-namemod', 'E-nationality', 'E-nickname', 'E-nicknametitle', 'E-norp:others', 'E-norp:political', 'E-ocean', 'E-org:edu', 'E-org:other', 'E-org:political', 'E-org:religious', 'E-orgcorp', 'E-periodic', 'E-person', 'E-port', 'E-product:drug', 'E-product:food', 'E-province', 'E-psudoname', 'E-quantity', 'E-rel', 'E-religion', 'E-restaurant', 'E-river', 'E-roadname', 'E-role', 'E-sciname', 'E-soi', 'E-song', 'E-space', 'E-sports_event', 'E-state', 'E-station', 'E-sub_district', 'E-time', 'E-title', 'E-tv_show', 'E-unit', 'E-vehicle', 'E-weapon', 'E-woa', 'E-year', 'I-address', 'I-animal_species', 'I-animate', 'I-army', 'I-book', 'I-bridge', 'I-building', 'I-cardinal', 'I-city', 'I-continent', 'I-country', 'I-date', 'I-day', 'I-disease', 'I-district', 'I-duration', 'I-electronics', 'I-event:others', 'I-facility:other', 'I-firstname', 'I-fund', 'I-goverment', 'I-hospital', 'I-hotel', 'I-island', 'I-jargon', 'I-last', 'I-law', 'I-loc:others', 'I-media', 'I-middle', 'I-month', 'I-namemod', 'I-norp:others', 'I-norp:political', 'I-org:edu', 'I-org:other', 'I-org:political', 'I-org:religious', 'I-orgcorp', 'I-periodic', 'I-person', 'I-port', 'I-product:food', 'I-province', 'I-psudoname', 'I-quantity', 'I-religion', 'I-restaurant', 'I-river', 'I-roadname', 'I-role', 'I-sciname', 'I-soi', 'I-song', 'I-sub_district', 'I-time', 'I-title', 'I-unit', 'I-vehicle', 'I-weapon', 'I-woa', 'I-year', 'O', 'S-address', 'S-airport', 'S-animal_species', 'S-animate', 'S-army', 'S-award', 'S-band', 'S-book', 'S-bridge', 'S-cardinal', 'S-city', 'S-continent', 'S-country', 'S-date', 'S-day', 'S-disease', 'S-distance', 'S-district', 'S-duration', 'S-electronics', 'S-event:others', 'S-facility:other', 'S-firstname', 'S-food:ingredient', 'S-fund', 'S-god', 'S-goverment', 'S-hospital', 'S-island', 'S-jargon', 'S-language', 'S-last', 'S-law', 'S-loc:others', 'S-media', 'S-middle', 'S-money', 'S-month', 'S-mountian', 'S-mult', 'S-namemod', 'S-nationality', 'S-natural_disaster', 'S-nickname', 'S-nicknametitle', 'S-norp:others', 'S-norp:political', 'S-ocean', 'S-org:edu', 'S-org:other', 'S-org:political', 'S-org:religious', 'S-orgcorp', 'S-percent', 'S-periodic', 'S-person', 'S-port', 'S-postcode', 'S-product:food', 'S-province', 'S-quantity', 'S-rel', 'S-religion', 'S-restaurant', 'S-river', 'S-roadname', 'S-role', 'S-sciname', 'S-soi', 'S-space', 'S-sports_event', 'S-state', 'S-station', 'S-stock_exchange', 'S-sub_district', 'S-temperature', 'S-time', 'S-title', 'S-unit', 'S-vehicle', 'S-weapon', 'S-weight', 'S-year']
    _LAYER_4_TAGS =  ['B-army', 'B-book', 'B-building', 'B-cardinal', 'B-city', 'B-continent', 'B-country', 'B-date', 'B-day', 'B-disease', 'B-district', 'B-duration', 'B-event:others', 'B-facility:other', 'B-firstname', 'B-fund', 'B-goverment', 'B-hotel', 'B-island', 'B-last', 'B-law', 'B-loc:others', 'B-media', 'B-middle', 'B-month', 'B-mountian', 'B-namemod', 'B-norp:others', 'B-norp:political', 'B-org:edu', 'B-org:other', 'B-org:political', 'B-orgcorp', 'B-person', 'B-port', 'B-product:food', 'B-province', 'B-quantity', 'B-religion', 'B-roadname', 'B-role', 'B-soi', 'B-state', 'B-sub_district', 'B-title', 'B-unit', 'B-year', 'E-army', 'E-book', 'E-building', 'E-cardinal', 'E-city', 'E-continent', 'E-country', 'E-date', 'E-day', 'E-disease', 'E-district', 'E-duration', 'E-event:others', 'E-facility:other', 'E-firstname', 'E-fund', 'E-goverment', 'E-hotel', 'E-island', 'E-last', 'E-law', 'E-loc:others', 'E-media', 'E-middle', 'E-month', 'E-mountian', 'E-namemod', 'E-norp:others', 'E-norp:political', 'E-org:edu', 'E-org:other', 'E-org:political', 'E-orgcorp', 'E-person', 'E-port', 'E-product:food', 'E-province', 'E-quantity', 'E-religion', 'E-roadname', 'E-role', 'E-soi', 'E-state', 'E-sub_district', 'E-title', 'E-unit', 'E-year', 'I-book', 'I-building', 'I-cardinal', 'I-city', 'I-country', 'I-date', 'I-day', 'I-disease', 'I-district', 'I-duration', 'I-event:others', 'I-facility:other', 'I-firstname', 'I-fund', 'I-goverment', 'I-hotel', 'I-last', 'I-law', 'I-loc:others', 'I-media', 'I-namemod', 'I-norp:others', 'I-norp:political', 'I-org:edu', 'I-org:other', 'I-org:political', 'I-orgcorp', 'I-port', 'I-product:food', 'I-province', 'I-quantity', 'I-roadname', 'I-role', 'I-soi', 'I-state', 'I-sub_district', 'I-title', 'I-unit', 'I-year', 'O', 'S-army', 'S-cardinal', 'S-city', 'S-continent', 'S-country', 'S-date', 'S-day', 'S-disease', 'S-district', 'S-electronics', 'S-event:others', 'S-facility:other', 'S-firstname', 'S-god', 'S-goverment', 'S-hospital', 'S-island', 'S-jargon', 'S-last', 'S-law', 'S-loc:others', 'S-media', 'S-money', 'S-month', 'S-mountian', 'S-mult', 'S-nationality', 'S-nickname', 'S-nicknametitle', 'S-norp:others', 'S-norp:political', 'S-ocean', 'S-org:edu', 'S-org:other', 'S-org:political', 'S-orgcorp', 'S-person', 'S-port', 'S-province', 'S-quantity', 'S-religion', 'S-river', 'S-roadname', 'S-role', 'S-soi', 'S-state', 'S-sub_district', 'S-time', 'S-title', 'S-unit', 'S-vehicle', 'S-war', 'S-weapon', 'S-year']
    _LAYER_5_TAGS =  ['B-country', 'B-disease', 'B-district', 'B-duration', 'B-firstname', 'B-goverment', 'B-law', 'B-loc:others', 'B-month', 'B-mountian', 'B-org:edu', 'B-org:other', 'B-org:political', 'B-orgcorp', 'B-religion', 'B-soi', 'B-sub_district', 'E-country', 'E-disease', 'E-district', 'E-duration', 'E-firstname', 'E-goverment', 'E-law', 'E-loc:others', 'E-month', 'E-mountian', 'E-org:edu', 'E-org:other', 'E-org:political', 'E-orgcorp', 'E-religion', 'E-soi', 'E-sub_district', 'I-disease', 'I-district', 'I-duration', 'I-firstname', 'I-law', 'I-loc:others', 'I-org:edu', 'I-org:other', 'I-org:political', 'I-orgcorp', 'I-sub_district', 'O', 'S-cardinal', 'S-continent', 'S-country', 'S-day', 'S-district', 'S-facility:other', 'S-firstname', 'S-goverment', 'S-hospital', 'S-law', 'S-loc:others', 'S-month', 'S-nationality', 'S-natural_disaster', 'S-norp:others', 'S-norp:political', 'S-org:edu', 'S-org:other', 'S-orgcorp', 'S-province', 'S-religion', 'S-roadname', 'S-role', 'S-sub_district', 'S-title', 'S-unit', 'S-year']
    _LAYER_6_TAGS =  ['B-disease', 'B-orgcorp', 'B-province', 'B-sub_district', 'E-disease', 'E-orgcorp', 'E-province', 'E-sub_district', 'I-orgcorp', 'I-province', 'O', 'S-cardinal', 'S-country', 'S-electronics', 'S-law', 'S-loc:others', 'S-month', 'S-nationality', 'S-norp:others', 'S-org:edu', 'S-province', 'S-religion', 'S-state', 'S-year']
    _LAYER_7_TAGS =  ['O', 'S-district', 'S-province', 'S-roadname']
    _LAYER_8_TAGS =  ['O', 'S-loc:others']

    def _info(self):
        return DatasetInfo(
            features=Features({
                "tokens": Sequence(Value("string")),
                "layer_1": Sequence(ClassLabel(names=self._LAYER_1_TAGS)),
                "layer_2": Sequence(ClassLabel(names=self._LAYER_2_TAGS)),
                "layer_3": Sequence(ClassLabel(names=self._LAYER_3_TAGS)),
                "layer_4": Sequence(ClassLabel(names=self._LAYER_4_TAGS)),
                "layer_5": Sequence(ClassLabel(names=self._LAYER_5_TAGS)),
                "layer_6": Sequence(ClassLabel(names=self._LAYER_6_TAGS)),
                "layer_7": Sequence(ClassLabel(names=self._LAYER_7_TAGS)),
                "layer_8": Sequence(ClassLabel(names=self._LAYER_8_TAGS)),
            })
        )

    def _split_generators(self, dl_manager):
        if not dl_manager.manual_dir:
            raise ValueError(
                "`data_dir` not provided. "
                "Make sure you insert a manual dir via "
                "`datasets.load_dataset('thai_nner', data_dir=...)`."
            )
        data_dir = path.abspath(path.expanduser(dl_manager.manual_dir))

        # check if manual folder exists
        if not path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} does not exist.")

        # check number of .conll files
        train_path = path.join(data_dir, self._TRAIN_FOLDER)
        test_path = path.join(data_dir, self._TEST_FOLDER)
        num_train = len(glob(path.join(train_path, "*.conll")))
        num_test = len(glob(path.join(test_path, "*.conll")))
        if num_train == 0:
            raise FileNotFoundError(f"No .conll files found in {train_path}.")
        if num_test == 0:
            raise FileNotFoundError(f"No .conll files found in {test_path}.")

        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"filepath": train_path},
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"filepath": test_path},
            )
        ]

    def _generate_examples(self, filepath):
        for file_idx, fname in enumerate(sorted(glob(path.join(filepath, "*.conll")))):
            with open(fname, encoding="utf-8") as f:
                guid = 0
                tokens = []
                layer_1_tags = []
                layer_2_tags = []
                layer_3_tags = []
                layer_4_tags = []
                layer_5_tags = []
                layer_6_tags = []
                layer_7_tags = []
                layer_8_tags = []

                for line in f:
                    if line == self._SENTENCE_SPLITTER:
                        if tokens:
                            yield f"{file_idx}_{guid}", {
                                "tokens": tokens,
                                "layer_1": layer_1_tags,
                                "layer_2": layer_2_tags,
                                "layer_3": layer_3_tags,
                                "layer_4": layer_4_tags,
                                "layer_5": layer_5_tags,
                                "layer_6": layer_6_tags,
                                "layer_7": layer_7_tags,
                                "layer_8": layer_8_tags,
                            }
                            guid += 1
                            tokens = []
                            layer_1_tags = []
                            layer_2_tags = []
                            layer_3_tags = []
                            layer_4_tags = []
                            layer_5_tags = []
                            layer_6_tags = []
                            layer_7_tags = []
                            layer_8_tags = []
                    else:
                        token, layer_1_tag, layer_2_tag, layer_3_tag, layer_4_tag, layer_5_tag, layer_6_tag, layer_7_tag, layer_8_tag = line.split()
                        tokens.append(token)
                        layer_1_tags.append(layer_1_tag)
                        layer_2_tags.append(layer_2_tag)
                        layer_3_tags.append(layer_3_tag)
                        layer_4_tags.append(layer_4_tag)
                        layer_5_tags.append(layer_5_tag)
                        layer_6_tags.append(layer_6_tag)
                        layer_7_tags.append(layer_7_tag)
                        layer_8_tags.append(layer_8_tag)
                # last example
                if tokens:
                    yield f"{file_idx}_{guid}", {
                        "tokens": tokens,
                        "layer_1": layer_1_tags,
                        "layer_2": layer_2_tags,
                        "layer_3": layer_3_tags,
                        "layer_4": layer_4_tags,
                        "layer_5": layer_5_tags,
                        "layer_6": layer_6_tags,
                        "layer_7": layer_7_tags,
                        "layer_8": layer_8_tags
                    }
