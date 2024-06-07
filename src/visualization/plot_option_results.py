import numpy as np
import matplotlib.pyplot as plt

def plot_option_prices_T(
        ground_paths_df, gen_paths_df, exact_call_put, call_payoff, put_payoff, r, T, t=0., grid_size=10, title_prefix = "European "
    ):
    # Store results
    T_values = []
    exact_call_prices = []
    exact_put_prices = []
    mc_call_ground_prices = []
    mc_put_ground_prices = []
    mc_call_gen_prices = []
    mc_put_gen_prices = []

    for i, T_cur in enumerate(ground_paths_df.columns):
        if i % grid_size != 0 and T_cur != T:
            continue        
        exact_call_price, exact_put_price = exact_call_put(T_cur)

        mc_call_ground_price = np.mean(np.exp(-r*(T_cur-t))*call_payoff(ground_paths_df.iloc[:, :(i+1)]))
        mc_put_ground_price = np.mean(np.exp(-r*(T_cur-t))*put_payoff(ground_paths_df.iloc[:, :(i+1)]))
        mc_call_gen_price = np.mean(np.exp(-r*(T_cur-t))*call_payoff(gen_paths_df.iloc[:, :(i+1)]))
        mc_put_gen_price = np.mean(np.exp(-r*(T_cur-t))*put_payoff(gen_paths_df.iloc[:, :(i+1)]))
        
        T_values.append(T_cur)
        exact_call_prices.append(exact_call_price)
        exact_put_prices.append(exact_put_price)
        mc_call_ground_prices.append(mc_call_ground_price)
        mc_put_ground_prices.append(mc_put_ground_price)
        mc_call_gen_prices.append(mc_call_gen_price)
        mc_put_gen_prices.append(mc_put_gen_price)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(T_values, exact_call_prices, 'b', label='Theoretical Call Price')
    plt.plot(T_values, mc_call_ground_prices, 'b--', label='Ground Truth Call Price')
    plt.plot(T_values, mc_call_gen_prices, 'b:', label='Generated Call Price')
    plt.plot(T_values, exact_put_prices, 'r', label='Theoretical Put Price')
    plt.plot(T_values, mc_put_ground_prices, 'r--', label='Ground Truth Put Price')
    plt.plot(T_values, mc_put_gen_prices, 'r:', label='Generated Put Price')

    plt.xlabel('Maturity (T)')
    plt.ylabel('Option Price')
    plt.title(title_prefix + 'Option Prices for Different Maturities (T)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return exact_call_prices, exact_put_prices, mc_call_ground_prices, mc_put_ground_prices, mc_call_gen_prices, mc_put_gen_prices, T_values

def plot_option_prices_K(
        ground_paths_df, gen_paths_df, K_values, exact_call_put, call_payoff, put_payoff, r, T, t=0, title_prefix = "European "
    ):
    # Store results
    exact_call_prices = []
    exact_put_prices = []
    mc_call_ground_prices = []
    mc_put_ground_prices = []
    mc_call_gen_prices = []
    mc_put_gen_prices = []

    for K in K_values:
        exact_call_price, exact_put_price = exact_call_put(K)

        mc_call_ground_price = np.mean(np.exp(-r*(T-t))*call_payoff(ground_paths_df, K))
        mc_put_ground_price = np.mean(np.exp(-r*(T-t))*put_payoff(ground_paths_df, K))
        mc_call_gen_price = np.mean(np.exp(-r*(T-t))*call_payoff(gen_paths_df, K))
        mc_put_gen_price = np.mean(np.exp(-r*(T-t))*put_payoff(gen_paths_df, K))

        exact_call_prices.append(exact_call_price)
        exact_put_prices.append(exact_put_price)
        mc_call_ground_prices.append(mc_call_ground_price)
        mc_put_ground_prices.append(mc_put_ground_price)
        mc_call_gen_prices.append(mc_call_gen_price)
        mc_put_gen_prices.append(mc_put_gen_price)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(K_values, exact_call_prices, 'b', label='Theoretical Call Price')
    plt.plot(K_values, mc_call_ground_prices, 'b--', label='Ground Truth Call Price')
    plt.plot(K_values, mc_call_gen_prices, 'b:', label='Generated Call Price')
    plt.plot(K_values, exact_put_prices, 'r', label='Theoretical Put Price')
    plt.plot(K_values, mc_put_ground_prices, 'r--', label='Ground Truth Put Price')
    plt.plot(K_values, mc_put_gen_prices, 'r:', label='Generated Put Price')

    plt.xlabel('Strike Price (K)')
    plt.ylabel('Option Price')
    plt.title(title_prefix + 'Option Prices for Different Strike Prices (K)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return exact_call_prices, exact_put_prices, mc_call_ground_prices, mc_put_ground_prices, mc_call_gen_prices, mc_put_gen_prices

def plot_option_price_deviation_absolute(
        mc_call_ground_prices, mc_put_ground_prices, mc_call_gen_prices, mc_put_gen_prices, exact_call_prices, exact_put_prices, x_values, x_label = "Strike Price (K)",  title_prefix = "European "
    ):
    # Store results
    ground_call_deviations = abs(np.array(mc_call_ground_prices) - np.array(exact_call_prices))
    ground_put_deviations = abs(np.array(mc_put_ground_prices) - np.array(exact_put_prices))
    gen_call_deviations = abs(np.array(mc_call_gen_prices) - np.array(exact_call_prices))
    gen_put_deviations = abs(np.array(mc_put_gen_prices) - np.array(exact_put_prices))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, ground_call_deviations, 'b--' ,label='Ground Truth Call Deviation')
    plt.plot(x_values, gen_call_deviations, 'b:' ,label='Generated Call Deviation')
    plt.plot(x_values, ground_put_deviations, 'r--',label='Ground Truth Put Deviation')
    plt.plot(x_values, gen_put_deviations, 'r:',label='Generated Put Deviation')

    plt.xlabel(x_label)
    plt.ylabel('Deviation from Theoretical Price')
    plt.title(f'Deviation of {title_prefix}Option Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

    return ground_call_deviations, ground_put_deviations, gen_call_deviations, gen_put_deviations


def plot_option_price_deviation_relative(
        ground_call_deviations, ground_put_deviations,  gen_call_deviations, gen_put_deviations, exact_call_prices, exact_put_prices, x_values, x_label = "Strike Price (K)", title_prefix = "European ", zoom_ylimits=(0, 10)
    ):

    ground_call_deviations_rel = ground_call_deviations / np.array(exact_call_prices) * 100
    ground_put_deviations_rel = ground_put_deviations / np.array(exact_put_prices) * 100
    gen_call_deviations_rel = gen_call_deviations / np.array(exact_call_prices) * 100
    gen_put_deviations_rel = gen_put_deviations / np.array(exact_put_prices) * 100

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(x_values, ground_call_deviations_rel, 'b--', label='Ground Truth Call Deviation')
    plt.plot(x_values, gen_call_deviations_rel, 'b:', label='Generated Call Deviation')
    plt.plot(x_values, ground_put_deviations_rel, 'r--', label='Ground Truth Put Deviation')
    plt.plot(x_values, gen_put_deviations_rel, 'r:', label='Generated Put Deviation')

    plt.xlabel(x_label)
    plt.ylabel('Relative Deviation from Theoretical Price (%)')
    plt.title(f'Relative Deviation of {title_prefix}Option Prices')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Different y-axis
    plt.figure(figsize=(10, 6))

    plt.plot(x_values, ground_call_deviations_rel, 'b--', label='Ground Truth Call Deviation')
    plt.plot(x_values, gen_call_deviations_rel, 'b:', label='Generated Call Deviation')
    plt.plot(x_values, ground_put_deviations_rel, 'r--', label='Ground Truth Put Deviation')
    plt.plot(x_values, gen_put_deviations_rel, 'r:', label='Generated Put Deviation')

    plt.xlabel(x_label)
    plt.ylabel('Relative Deviation from Theoretical Price (%)')
    plt.title(f'Relative Deviation of {title_prefix}Option Prices')
    plt.legend()
    plt.ylim(zoom_ylimits)
    plt.grid(True)
    plt.show()

    return ground_call_deviations_rel, ground_put_deviations_rel, gen_call_deviations_rel, gen_put_deviations_rel



