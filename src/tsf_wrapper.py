# ***********************************************************************
# * Licensed Materials - Property of IBM
# *
# * IBM SPSS Products: Statistics Common
# *
# * (C) Copyright IBM Corp. 1989, 2025
# *
# * US Government Users Restricted Rights - Use, duplication or disclosure
# * restricted by GSA ADP Schedule Contract with IBM Corp.
# ************************************************************************

from wrapper.basewrapper import *
from wrapper import wraputil
from util.statjson import *

import numpy as np
import statsmodels.api as sm
from datetime import datetime
import re

import warnings
import traceback

import logging
logging.basicConfig(level=logging.ERROR)
log_error = logging.error 

warnings.simplefilter("error", category=RuntimeWarning)

"""Initialize the tsf wrapper"""
init_wrapper("tsf", os.path.join(os.path.dirname(__file__), "TSF-properties.json"))

hp_filter = True
bk_filter = False
cf_filter = False


def execute(iterator_id, data_model, settings, lang="en"):
    fields = data_model["fields"]
    output_json = StatJSON(get_name())
    xtIntl = get_lang_resource(lang)

    check_settings(settings, fields)

    date_factor = get_value("factors")
    time_variable_index = wraputil.get_index(fields, "DATE_")

    #check if date_factor is set and has values, if not check if time_variable_index is set and not -1
    #if both are not set, log warning and return.
    if (date_factor is None or len(date_factor) == 0) and (time_variable_index is None or time_variable_index == -1):
        log_error(xtIntl.loadstring("no_date_factor"))
        warning_item = Warnings(xtIntl.loadstring("no_time_var_error"))
        output_json.add_warnings(warning_item)
        return
    
    def execute_model(data):
            
        if data is not None:
            case_count = len(data)
        else:
            return
        try:
            global hp_filter, bk_filter, cf_filter

            records = RecordData(data)
            columns_data = records.get_columns()
            no_of_rows = len(data)

            time_data = []

            if date_factor is not None and len(date_factor) > 0:
                time_data = parse_and_sort_factors(date_factor)
            else:
                time_data = columns_data[time_variable_index]

            hp_filter = get_value("hpfilter")
            bk_filter = get_value("bkfilter")
            cf_filter = get_value("cffilter")

            hp_variable = get_value("hpvariable")

            lamda = get_value("lamb")
            if lamda is None or lamda == "":
                lamda = 1600

            bk_cf_names = []
            bk_cf_fnotes = []

            if is_set("bk_cf_variables"):
                 bk_cf_names.extend(get_value("bk_cf_variables"))
                 bk_cf_fnotes.extend(get_value("bk_cf_variables"))
                
            result = {}

            if hp_filter:
                # Adding here logic for hp_filter
                result["hp_filter"] = {}

                hp_index = wraputil.get_index(fields, hp_variable)
                if hp_index is not None:
                    hp_data = columns_data[hp_index]
                    ts_data = np.array(hp_data)

                    if np.any(np.isnan(ts_data)) or np.any(np.isinf(ts_data)):
                        raise ValueError(f"{hp_variable} contains NaN or inf values, cannot apply HP filter.")
                    
                    cycle, trend = sm.tsa.filters.hpfilter(ts_data, lamda)
                    result["hp_filter"] = {
                        "cycle": cycle.tolist(),
                        "trend": trend.tolist()
                    }
                    create_hp_filter_output(xtIntl, result, time_data, hp_data, hp_variable, fields, output_json)

            if bk_filter or cf_filter:
                if is_set("bk_cf_variables"):
                    for var in bk_cf_names:
                        var_index = wraputil.get_index(fields, var)
                        if var_index is not None:
                            var_data = columns_data[var_index]
                            create_bk_cf_variable_time_series_plot(xtIntl, var_data, time_data, var, fields, output_json)


            if bk_filter:
                # Adding here logic for bk_filter
                if is_set("bk_cf_variables"):
                    low = get_value("low") or 6.0
                    high = get_value("high") or 32.0
                    k = get_value("k") or 12
                    k = int(k)

                    if low >= high:
                        raise ValueError(f"low ({low}) must be less than high ({high})")

                    if k <= 0:
                        raise ValueError(f"K ({k}) must be a positive integer")

                    result["bk_filter"] = {}

                    for var in bk_cf_names:
                        try:
                            var_index = wraputil.get_index(fields, var)
                            if var_index is not None:
                                var_data = columns_data[var_index]
                                ts_data = np.array(var_data)

                                if len(ts_data) < 3:
                                    log_error(f"Variable '{var}': Needs ≥3 data points")
                                    continue
                                if np.isnan(ts_data).any():
                                    log_error(f"Variable '{var}': Contains NaN")
                                    continue
                                bk_filter_data = sm.tsa.filters.bkfilter(ts_data, low, high, K=k)
                                result["bk_filter"][var] = bk_filter_data.tolist()
                        except Exception as e:
                            error_msg = f"Error processing variable '{var}': {str(e)}"
                            log_error(error_msg)
                            raise type(e)(error_msg) from e
                    create_bk_filter_output(xtIntl, result, time_data, bk_filter_data, fields, output_json, bk_cf_names)


            if cf_filter:
                if is_set("bk_cf_variables"):
   
                    low = get_value("low")
                    low = 6.0 if low is None else low
                    high = get_value("high")
                    high = 32.0 if high is None else high
                    drift = get_value("drift")

                    if low >= high:
                        raise ValueError(f"low ({low}) must be less than high ({high})")

                    result["cf_filter"] = {}
                    
                    for var in bk_cf_names:
                        var_index = wraputil.get_index(fields, var)
                        if var_index is not None:
                            ts_data = np.array(columns_data[var_index])
                            if len(ts_data) < 3:
                                raise ValueError(f"Insufficient data points for {var}.")
                            
                            if np.any(np.isnan(ts_data)) or np.any(np.isinf(ts_data)):
                                raise ValueError(f"{var} contains NaN or inf values, cannot apply CF filter.")
                            
                            if drift is not None:
                                cycle, trend = sm.tsa.filters.cffilter(ts_data, low, high, drift=drift)
                            else:
                                cycle, trend = sm.tsa.filters.cffilter(ts_data, low, high)

                            result["cf_filter"][var] = {
                                "cycle": cycle.tolist(),
                                "trend": trend.tolist()
                            }
                    create_cf_filter_output(xtIntl, result, time_data, fields, output_json, bk_cf_names)

        except Exception as err:
            warning_item = Warnings(xtIntl.loadstring("python_returned_msg") + "\n" + repr(err))
            output_json.add_warnings(warning_item)
            tb = traceback.format_exc()

            notes = Notes(xtIntl.loadstring("python_output"), tb)
            output_json.add_notes(notes)
        finally:
            generate_output(output_json.get_json(), None)
            finish()

    get_records(iterator_id, data_model, execute_model)
    return 0


def create_hp_filter_output(xtIntl, result, time_data, hp_data, hp_variable, fields, output_json):
    
    # HP variable vs Time Data Plot
    chart_title = xtIntl.loadstring("hp_variable_chart_title")+ f" {hp_variable}"
    chart_x_label = xtIntl.loadstring("date_label")
    chart_x_data = list(time_data)
    chart_y_label = hp_variable
    chart_y_data = hp_data

    graph_dataset = "hp_filter_graph_data"
    
    hp_variable_chart = GplChart(chart_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(chart_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(chart_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    hp_variable_chart.add_gpl_statement(gpl_statements)
    hp_variable_chart.add_variable_mapping("x", chart_x_data, graph_dataset)
    hp_variable_chart.add_variable_mapping("y", chart_y_data, graph_dataset)

    output_json.add_chart(hp_variable_chart)
    

    #HP Filter Trends vs Time plot
    hp_filter_results = result["hp_filter"]
    cycle = hp_filter_results["cycle"]
    trend = hp_filter_results["trend"]

    trend_title = xtIntl.loadstring("hp_filter_trend_title")
    trend_x_label = xtIntl.loadstring("date_label")
    trend_x_data = list(time_data)
    trend_y_label = xtIntl.loadstring("trend_label")

    graph_dataset = "hp_filter_trend_data"

    hp_filter_trend_chart = GplChart(trend_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(trend_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(trend_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(trend_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    hp_filter_trend_chart.add_gpl_statement(gpl_statements)
    hp_filter_trend_chart.add_variable_mapping("x", trend_x_data, graph_dataset)
    hp_filter_trend_chart.add_variable_mapping("y", trend, graph_dataset)

    output_json.add_chart(hp_filter_trend_chart)

    #HP Filter Cycle vs Time plot
    cycle_title = xtIntl.loadstring("hp_filter_cycle_title")
    cycle_x_label = xtIntl.loadstring("date_label")
    cycle_x_data = list(time_data)
    cycle_y_label = xtIntl.loadstring("cycle_label")

    graph_dataset = "hp_filter_cycle_data"

    hp_filter_cycle_chart = GplChart(cycle_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(cycle_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(cycle_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(cycle_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    hp_filter_cycle_chart.add_gpl_statement(gpl_statements)
    hp_filter_cycle_chart.add_variable_mapping("x", cycle_x_data, graph_dataset)
    hp_filter_cycle_chart.add_variable_mapping("y", cycle, graph_dataset)

    output_json.add_chart(hp_filter_cycle_chart)

    #HP Filter Trend and Variable combine Plot
    title = xtIntl.loadstring("hp_filter_trend_and_variable_plot")
    hpvar_trend_chart_title = f"{title} {hp_variable}"
    hpvar_trend_x_label = xtIntl.loadstring("date_label")
    hpvar_trend_y_label = xtIntl.loadstring("combined_y_data_label")
    subfootnote = f"Source: {hp_variable}"


    x_data = list(time_data)
    hpvar_trend_x_data = x_data * 2  
    hpvar_trend_y_data = trend + hp_data

    color_data = (["Trend"] * len(time_data)) + ([hp_variable] * len(time_data))

    graph_dataset = "hp_filter_combined_graph_data"

    gpl_chart = GplChart(hpvar_trend_chart_title)

    gpl_statements = [
        "SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x=col(source(s), name(\"x\"), unit.category())",
        "DATA: y=col(source(s), name(\"y\"))",
        "DATA: color=col(source(s), name(\"color\"), unit.category())",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(hpvar_trend_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(hpvar_trend_y_label),
        "GUIDE: legend(aesthetic(aesthetic.color.interior))",
        "GUIDE: text.title(label(\"{0}\"))".format(hpvar_trend_chart_title),
        "GUIDE: text.subfootnote(label(\"{0}\"))".format(subfootnote),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "SCALE: cat(aesthetic(aesthetic.color.interior))",
        "ELEMENT: line(position(x*y), color.interior(color), size(size.\"1pt\"))",
       #"ELEMENT: point(position(x*y), color.interior(color), size(size.\"3pt\"))"
    ]

    gpl_chart.add_gpl_statement(gpl_statements)


    gpl_chart.add_variable_mapping("x", hpvar_trend_x_data, graph_dataset)
    gpl_chart.add_variable_mapping("y", hpvar_trend_y_data, graph_dataset)
    gpl_chart.add_variable_mapping("color", color_data, graph_dataset)


    output_json.add_chart(gpl_chart)


def create_bk_filter_output(xtIntl, result, time_data, bk_filter_data, fields, output_json, bk_cf_names):

    size_of_bk_cf_field = len(bk_cf_names)
    var1 = bk_cf_names[0]

    #BK variable-1 vs Time Data Plot

    title = xtIntl.loadstring("bk_filter_plot_title")
    var1_chart_title = title + f" {var1}"
    var1_bk_x_label = xtIntl.loadstring("date_label")
    var1_bk_x_data = list(time_data)
    var1_bk_y_label = var1
    var1_bk_y_data = result["bk_filter"][var1]

    graph_dataset = f"bk_filter_graph_{var1}_data"

    var1_bk_chart = GplChart(var1_chart_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(var1_bk_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(var1_bk_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(var1_chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    var1_bk_chart.add_gpl_statement(gpl_statements)
    var1_bk_chart.add_variable_mapping("x", var1_bk_x_data, graph_dataset)
    var1_bk_chart.add_variable_mapping("y", var1_bk_y_data, graph_dataset)

    output_json.add_chart(var1_bk_chart)

    if size_of_bk_cf_field < 2:
        return
    
    var2 = bk_cf_names[1]
    
    #BK variable-2 vs Time Data Plot
    var2_chart_title = title + f" {var2}"
    var2_bk_x_label = xtIntl.loadstring("date_label")
    var2_bk_x_data = list(time_data)
    var2_bk_y_label = var2
    var2_bk_y_data = result["bk_filter"][var2]

    graph_dataset = f"bk_filter_graph_{var2}_data"

    var2_bk_chart = GplChart(var2_chart_title)  

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(var2_bk_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(var2_bk_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(var2_chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    var2_bk_chart.add_gpl_statement(gpl_statements)
    var2_bk_chart.add_variable_mapping("x", var2_bk_x_data, graph_dataset)
    var2_bk_chart.add_variable_mapping("y", var2_bk_y_data, graph_dataset)
  
    output_json.add_chart(var2_bk_chart)

    #BK Filter combined Plot for Variable 1 and Variable 2
    title_combined = xtIntl.loadstring("bk_filter_plot_combined_title")
    combined_chart_title = f"{title_combined} {var1} & {var2}"
    combined_x_label = xtIntl.loadstring("date_label")
    combined_y_label = xtIntl.loadstring("combined_y_data_label")
    combined_subfootnote = f"Source: {var1} and {var2}"
    
    minimum_length = min(len(time_data), len(var1_bk_y_data), len(var2_bk_y_data))

    time_data = time_data[:minimum_length]
    var1_bk_y_data = var1_bk_y_data[:minimum_length]
    var2_bk_y_data = var2_bk_y_data[:minimum_length]

    combined_x_data = list(time_data) * 2
    combined_y_data = var1_bk_y_data + var2_bk_y_data
    combined_color_data = [var1] * len(time_data) + [var2] * len(time_data)
    
 
    graph_dataset = "bk_filter_graph_data"
    
    gpl_chart = GplChart(combined_chart_title)
    
    gpl_statements = [
        "SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x=col(source(s), name(\"x\"), unit.category())",
        "DATA: y=col(source(s), name(\"y\"))",
        "DATA: color=col(source(s), name(\"color\"), unit.category())",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(combined_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(combined_y_label),
        "GUIDE: legend(aesthetic(aesthetic.color.interior))",
        "GUIDE: text.title(label(\"{0}\"))".format(combined_chart_title),
        "GUIDE: text.subfootnote(label(\"{0}\"))".format(combined_subfootnote),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "SCALE: cat(aesthetic(aesthetic.color.interior))",
        "ELEMENT: line(position(x*y), color.interior(color), size(size.\"1pt\"))",
       #"ELEMENT: point(position(x*y), color.interior(color), size(size.\"3pt\"))"
    ]
    
    gpl_chart.add_gpl_statement(gpl_statements)
    
    gpl_chart.add_variable_mapping("x", combined_x_data, graph_dataset)
    gpl_chart.add_variable_mapping("y", combined_y_data, graph_dataset)
    gpl_chart.add_variable_mapping("color", combined_color_data, graph_dataset)
    
    output_json.add_chart(gpl_chart)


def create_cf_filter_output(xtIntl, result, time_data, fields, output_json, bk_cf_names):

    size_of_bk_cf_field = len(bk_cf_names)

    var1 = bk_cf_names[0]

    var1_cycle_data = result["cf_filter"][var1]["cycle"]

    title_var1 = xtIntl.loadstring("cf_filter_plot_title")
    var1_cycle_chart_title = f"{title_var1} {var1}"
    var1_cycle_cf_x_label = xtIntl.loadstring("date_label")
    var1_cycle_cf_x_data = list(time_data)
    var1_cycle_cf_y_label = var1
    var1_cycle_cf_y_data = var1_cycle_data

    graph_dataset = f"cf_filter_graph_{var1}_data"

    var1_cf_chart = GplChart(var1_cycle_chart_title)
    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(var1_cycle_cf_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(var1_cycle_cf_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(var1_cycle_chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]
    var1_cf_chart.add_gpl_statement(gpl_statements)
    var1_cf_chart.add_variable_mapping("x", var1_cycle_cf_x_data, graph_dataset)
    var1_cf_chart.add_variable_mapping("y", var1_cycle_cf_y_data, graph_dataset)

    output_json.add_chart(var1_cf_chart)

    if size_of_bk_cf_field  < 2:
        return
    
    var2 = bk_cf_names[1]

    var2_cycle_data = result["cf_filter"][var2]["cycle"]


    title_var2 = xtIntl.loadstring("cf_filter_plot_title")
    var2_cycle_chart_title = f"{title_var2} {var2}"
    var2_cycle_cf_x_label = xtIntl.loadstring("date_label")
    var2_cycle_cf_x_data = list(time_data)
    var2_cycle_cf_y_label = var2
    var2_cycle_cf_y_data = var2_cycle_data

    graph_dataset = f"cf_filter_graph_{var2}_data"

    var2_cf_chart = GplChart(var2_cycle_chart_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(var2_cycle_cf_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(var2_cycle_cf_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(var2_cycle_chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    var2_cf_chart.add_gpl_statement(gpl_statements)
    var2_cf_chart.add_variable_mapping("x", var2_cycle_cf_x_data, graph_dataset)
    var2_cf_chart.add_variable_mapping("y", var2_cycle_cf_y_data, graph_dataset)

    output_json.add_chart(var2_cf_chart)

    combined_chart_title = xtIntl.loadstring("cf_filter_comparison_plot_title") + f" {var1} & {var2}"
    combined_x_label = xtIntl.loadstring("date_label")
    combined_y_label = xtIntl.loadstring("combined_y_data_label")
    combined_subfootnote = f"Source: {var1} and {var2}"
    combined_x_data = list(time_data) * 2
    combined_y_data = var1_cycle_data + var2_cycle_data
    combined_color_data = [var1] * len(time_data) + [var2] * len(time_data)

    graph_dataset = "cf_filter_graph_data"
    gpl_chart = GplChart(combined_chart_title)
    gpl_statements = [
        "SOURCE: s=userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x=col(source(s), name(\"x\"), unit.category())",
        "DATA: y=col(source(s), name(\"y\"))",
        "DATA: color=col(source(s), name(\"color\"), unit.category())",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(combined_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(combined_y_label),
        "GUIDE: legend(aesthetic(aesthetic.color.interior))",
        "GUIDE: text.title(label(\"{0}\"))".format(combined_chart_title),
        "GUIDE: text.subfootnote(label(\"{0}\"))".format(combined_subfootnote),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "SCALE: cat(aesthetic(aesthetic.color.interior))",
        "ELEMENT: line(position(x*y), color.interior(color), size(size.\"1pt\"))",
       #"ELEMENT: point(position(x*y), color.interior(color), size(size.\"3pt\"))"
    ]
    gpl_chart.add_gpl_statement(gpl_statements)
    gpl_chart.add_variable_mapping("x", combined_x_data, graph_dataset)
    gpl_chart.add_variable_mapping("y", combined_y_data, graph_dataset)
    gpl_chart.add_variable_mapping("color", combined_color_data, graph_dataset)

    output_json.add_chart(gpl_chart)

def create_bk_cf_variable_time_series_plot(xtIntl, var_data, time_data, var, fields, output_json):

    #BK and CF variable vs Time Data Plot
    chart_title = xtIntl.loadstring("hp_variable_chart_title") + f" {var}"
    chart_x_label = xtIntl.loadstring("date_label")
    chart_x_data = list(time_data)
    chart_y_label = var
    chart_y_data = var_data

    graph_dataset = f"bk_cf_graph_{var}_data"

    time_series_chart = GplChart(chart_title)

    gpl_statements = [
        "SOURCE: s = userSource(id(\"{0}\"))".format(graph_dataset),
        "DATA: x = col(source(s), name(\"x\"), unit.category())",
        "DATA: y = col(source(s), name(\"y\"))",
        "GUIDE: axis(dim(1), label(\"{0}\"))".format(chart_x_label),
        "GUIDE: axis(dim(2), label(\"{0}\"))".format(chart_y_label),
        "GUIDE: text.title(label(\"{0}\"))".format(chart_title),
        "SCALE: cat(dim(1), sort.data())",
        "SCALE: linear(dim(2))",
        "ELEMENT: line(position(x*y),size(size.\"1pt\"))",
    ]

    time_series_chart.add_gpl_statement(gpl_statements)
    time_series_chart.add_variable_mapping("x", chart_x_data, graph_dataset)
    time_series_chart.add_variable_mapping("y", chart_y_data, graph_dataset)

    output_json.add_chart(time_series_chart)


def parse_and_sort_factors(raw_factors):
    cleaned_dates = []
    for factor in raw_factors:
        match = re.search(r'DATE_=(.+?)\]', factor)
        if match:
            date_str = re.sub(r'\s+', ' ', match.group(1).strip())
            cleaned_dates.append(date_str)

    def is_quarter_format(date):
        return re.match(r'^Q[1-4] \d{4}$', date)

    def is_month_format(date):
        return re.match(r'^[A-Z]{3} \d{4}$', date)

    def is_weekday_format(date):
        return re.match(
            r'^(\d+\s+(?:SUN|MON|TUE|WED|THU|FRI|SAT)|'
            r'(?:SUN|MON|TUE|WED|THU|FRI|SAT)\s+\d+)$', 
            date
        )

    weekday_order = {
        'SUN': 0, 'MON': 1, 'TUE': 2, 'WED': 3,
        'THU': 4, 'FRI': 5, 'SAT': 6
    }

    def sort_key(date_str):
        if is_quarter_format(date_str):
            quarter, year = date_str.split()
            return int(year) * 4 + (int(quarter[1]) - 1)
        elif is_month_format(date_str):
            dt = datetime.strptime(date_str, "%b %Y")
            return dt.year * 12 + dt.month
        else:
            return float('inf')

    if cleaned_dates:
        if all(is_weekday_format(d) for d in cleaned_dates):
            first_part = cleaned_dates[0].split()[0]
            
            if first_part in weekday_order:
                sorted_dates = sorted(
                    cleaned_dates,
                    key=lambda x: (
                        weekday_order[x.split()[0]], 
                        int(x.split()[1])
                    )
                )
            else:
                sorted_dates = sorted(
                    cleaned_dates,
                    key=lambda x: (
                        int(x.split()[0]), 
                        weekday_order[x.split()[1]]
                    )
                )
        elif all(is_quarter_format(d) for d in cleaned_dates) or all(is_month_format(d) for d in cleaned_dates):
            sorted_dates = sorted(cleaned_dates, key=sort_key)
        else:
            sorted_dates = cleaned_dates
    else:
        sorted_dates = []

    return sorted_dates


