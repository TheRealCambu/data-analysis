# def plot_results_osnr(
#         data_dict: dict,
#         plot_ber: bool = True,
#         plot_ber_variance: bool = False,
#         plot_per_pol_ber: bool = False,
#         plot_evm: bool = False,
#         plot_evm_variance: bool = False,
#         plot_per_pol_evm: bool = False,
#         plot_ber_evm: bool = False,
#         plot_ber_evm_variance: bool = False,
#         plot_per_pol_ber_evm: bool = False,
#         fec_threshold: float = 2e-2
# ):
#     # TODO: Filter the data
#     # TODO: Put the FEC threshold
#
#     # Extract the data
#     baud_rate = np.unique(data_dict['symbol_rate'])
#     osnr = data_dict['osnr']
#     ber_tot = data_dict['ber_tot']
#     ber_x = data_dict['ber_x']
#     ber_y = data_dict['ber_y']
#     evm_tot = data_dict['evm_tot']
#     evm_x = data_dict['evm_x']
#     evm_y = data_dict['evm_y']
#     ber_evm_tot = data_dict['ber_evm_tot']
#     ber_evm_x = data_dict['ber_evm_x']
#     ber_evm_y = data_dict['ber_evm_y']
