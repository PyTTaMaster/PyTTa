.. currentmodule:: pytta

Moving from the ITA-Toolbox
===========================

This page is for users of the ITA-Toolbox for MATLAB (Institute of Technical
Acoustics, RWTH Aachen) who want to do the same work in PyTTa. It lists the
PyTTa counterpart of the ITA functions that have one, shows a few common
tasks side by side, and notes the conventions that differ.

The status column gives "in PyTTa" for what is in the package and the number
of the pull request for what is still under review; issue #75 lists those
pull requests and the order in which they build on each other.

Conventions that differ
-----------------------

* Indices start at 0. A start sample of 0 in PyTTa is sample 1 in the
  ITA-Toolbox.
* ``SignalObj`` takes the place of ``itaAudio``. ``timeSignal`` holds the
  samples, one column per channel, stored as float32; results compared with
  double precision agree to about 1e-7 of the signal scale.
* ``freqSignal`` follows the normalization of PyTTa for power and energy
  signals and is not the plain DFT of the samples. Functions that need the
  DFT compute ``numpy.fft.rfft(signal.timeSignal, axis=0)``.
* Options given in MATLAB as name and value pairs are keyword arguments in
  Python, e.g. ``ita_smooth(a, 'LogFreqOctave1', 1/3)`` becomes
  ``pytta.smooth_frequency(a, nthOct=3)``.
* The modules for sound fields (``pytta.analytic``, ``pytta.fem``,
  ``pytta.bem``) use the time dependence exp(jωt); check the convention of
  any other source before combining complex results.
* Where the ITA-Toolbox departs from a standard or from the paper it cites,
  PyTTa follows the reference and the docstring states the difference.

Common tasks side by side
-------------------------

Room acoustic parameters of a measured impulse response, in octave bands from
125 Hz to 4 kHz::

    % ITA-Toolbox
    ir = ita_read('ir.wav');
    res = ita_roomacoustics(ir, 'freqRange', [125 4000], ...
                            'bandsPerOctave', 1, 'T30', 'C80', 'EDT');
    res.T30.freq

    # PyTTa
    ir = pytta.read_wav('ir.wav')
    room = pytta.RoomAnalysis(ir, nthOct=1, minFreq=125, maxFreq=4000)
    room.T30, room.C80, room.EDT

Third octave smoothing of the magnitude spectrum::

    % ITA-Toolbox
    smooth = ita_smooth(ir, 'LogFreqOctave1', 1/3, 'Abs');

    # PyTTa
    smooth = pytta.smooth_frequency(ir, nthOct=3)

Third octave band filtering, one channel per band::

    % ITA-Toolbox
    bands = ita_fractional_octavebands(ir, 'bandsperoctave', 3, ...
                                       'freqRange', [100 5000]);

    # PyTTa
    bands = pytta.OctFilter(order=4, nthOct=3, samplingRate=ir.samplingRate,
                            minFreq=100, maxFreq=5000, refFreq=1000,
                            base=10).filter(ir)[0]

Loudness of a sound pressure signal in Pa (ISO 532-1, Zwicker)::

    % ITA-Toolbox (calls acousticLoudness of the Audio Toolbox)
    sig.channelUnits = {'Pa'};
    N = ita_psychoacoustic_loudness(sig);

    # PyTTa
    result = pytta.psychoacoustics.loudness_zwicker(sig, field='free')
    result['N'], result['LN']

A measurement without a sound card, for testing and teaching, with a known
system (``itaMSTFdummy`` in the ITA-Toolbox)::

    # PyTTa
    m = pytta.classes.SimulatedFRFMeasure(system=h, excitation=sweep,
                                          noise=-80., inChannels=[1],
                                          outChannels=[1], device=0)
    ir = m.run()          # an ImpulsiveResponse, as FRFMeasure.run() gives

Functions by topic
------------------

Signals and files
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_read, ita_read_wav
     - ``pytta.read_wav``, ``pytta.load``
     - in PyTTa
   * - ita_write, ita_write_wav
     - ``pytta.write_wav``, ``pytta.save``
     - in PyTTa
   * - ita_read_flac, ita_read_caf, ita_read_au
     - ``pytta.read_audio``, ``pytta.write_audio``
     - #104
   * - ita_read_blockwise
     - ``pytta.read_blockwise``
     - #104
   * - ita_write_txt
     - ``pytta.write_txt``, ``pytta.read_txt``
     - #104
   * - ita_read_sofa, ita_write_sofa
     - ``pytta.sofa.read``, ``pytta.sofa.write``
     - #83
   * - ita_merge, ita_split
     - ``pytta.merge``, ``pytta.split``
     - in PyTTa
   * - ita_resample, ita_oversample
     - ``pytta.resample``
     - in PyTTa
   * - ita_convolve, ita_fconv
     - ``pytta.fft_convolve``
     - in PyTTa
   * - ita_xcorr_dat
     - ``pytta.find_delay``, ``pytta.corr_coef``
     - in PyTTa

Signal generation
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_generate('sine', ...)
     - ``pytta.generate.sin``
     - in PyTTa
   * - ita_generate_sweep
     - ``pytta.generate.sweep``
     - in PyTTa
   * - ita_generate_impulse
     - ``pytta.generate.impulse``
     - in PyTTa
   * - ita_generate('pinknoise', ...)
     - ``pytta.generate.colored_noise``
     - in PyTTa
   * - ita_generate_mls
     - ``pytta.generate.mls``, ``pytta.mls_deconvolution``
     - #80
   * - ita_multitone
     - ``pytta.generate.multitone``
     - #105
   * - ita_stochastic_room_impulse_response
     - ``pytta.generate.stochastic_ir``
     - #109

Editing
-------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_time_window
     - ``pytta.time_window``
     - #49
   * - ita_envelope
     - ``pytta.envelope``
     - #96
   * - ita_start_IR
     - ``pytta.start_ir``
     - #96
   * - ita_time_shift
     - ``pytta.time_shift``
     - #96
   * - ita_append
     - ``pytta.append``
     - #96
   * - ita_normalize_dat, ita_normalize_spk
     - ``pytta.normalize``
     - #96
   * - ita_amplify_to
     - ``pytta.amplify_to``
     - #96
   * - ita_zerophase, ita_randomize_phase
     - ``pytta.zero_phase``, ``pytta.randomize_phase``
     - #96
   * - ita_quantize, ita_dither
     - ``pytta.quantize``, ``pytta.dither``
     - #96, #113
   * - ita_xfade_spk
     - ``pytta.crossfade_spectra``
     - #105
   * - ita_cutImpulseTrain
     - ``pytta.cut_impulse_train``
     - #105

Filters and smoothing
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_fractional_octavebands, ita_mpb_filter
     - ``pytta.OctFilter``
     - in PyTTa
   * - ita_filter_LiRi, ita_butterFR
     - ``pytta.butterworth``
     - #97
   * - ita_filter_notch, ita_filter_peak
     - ``pytta.notch``, ``pytta.peak_eq``
     - #97
   * - ita_filter_weighting
     - ``pytta.iec61672.weighting_filter``
     - #72
   * - ita_smooth
     - ``pytta.smooth_frequency``
     - #50
   * - ita_smooth_notches
     - ``pytta.smooth_notches``
     - #120
   * - ita_lifter
     - ``pytta.lifter``
     - #119
   * - ita_invert_spk_minphase_regularization
     - ``pytta.invert_spectrum``
     - #113
   * - ita_audio_warp, ita_zpk_warp, barkwarp
     - ``pytta.warp``, ``pytta.warp_zpk``, ``pytta.bark_warping``
     - #111
   * - ita_prony_analysis
     - ``pytta.prony``, ``pytta.invfreqz``
     - #110

Spectral analysis
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_spectrogram_mgu
     - ``pytta.plot_spectrogram``, ``pytta.enhanced_spectrogram``
     - #51
   * - ita_minimumphase
     - ``pytta.minimum_phase``
     - #56
   * - ita_groupdelay
     - ``pytta.group_delay``
     - #80
   * - ita_coherence
     - ``pytta.coherence``
     - #80
   * - ita_cepstrum, ita_icepstrum
     - ``pytta.cepstrum``, ``pytta.icepstrum``
     - #80
   * - ita_psd
     - ``pytta.psd``
     - #98
   * - ita_interpolate_spk
     - ``pytta.interpolate_spectrum``
     - #98
   * - ita_spk_extrapolate, ita_extrapolate
     - ``pytta.extrapolate_low``, ``pytta.extrapolate``
     - #98, #121
   * - ita_pitch_detection
     - ``pytta.pitch_detection``
     - #118
   * - ita_wiener_hopf_factorization
     - ``pytta.wiener_hopf``
     - #118
   * - ita_normxcorr2
     - ``pytta.normxcorr2``
     - #118
   * - ita_kurtosis
     - ``pytta.descriptors.kurtosis``
     - #73

Plots
-----

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_plot_time, ita_plot_time_dB
     - ``pytta.plot_time``, ``pytta.plot_time_dB``
     - in PyTTa
   * - ita_plot_freq, ita_plot_spk
     - ``pytta.plot_freq``
     - in PyTTa
   * - ita_plot_phase
     - ``pytta.plot_phase``
     - #68
   * - ita_plot_spectrogram
     - ``pytta.plot_spectrogram``
     - in PyTTa
   * - ita_plot_groupdelay, ita_plot_nyquist
     - ``pytta.plot_group_delay``, ``pytta.plot_nyquist``
     - #107
   * - ita_plot_zplane, ita_plot_polar
     - ``pytta.plot_zplane``, ``pytta.plot_polar``
     - #107

Measurement
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - itaMSRecord, itaMSPlaybackRecord, itaMSTF
     - ``pytta.RecMeasure``, ``pytta.PlayRecMeasure``, ``pytta.FRFMeasure``
     - in PyTTa
   * - itaMSTFdummy
     - ``pytta.classes.SimulatedFRFMeasure``
     - #117
   * - itaMSTFinterleaved, itaMSTFmimo
     - ``pytta.multiple_sweep_excitation``, ``pytta.multiple_sweep_irs``
     - #117
   * - ita_measurement_trigger
     - ``pytta.triggered_average``
     - #117
   * - ita_impulsiveNoiseDetection
     - ``pytta.impulsive_noise``
     - #113
   * - ita_add_nearfield_farfield_measurements
     - ``pytta.merge_near_far``
     - #113
   * - ita_nonlinear_extract_harmonics, ita_thd
     - ``pytta.harmonic_distortion``, ``pytta.sine_distortion``
     - #57, #64
   * - ita_nonlinear_power_to_harmonic_series
     - ``pytta.power_to_harmonic_series``, ``pytta.harmonic_to_power_series``
     - #108
   * - ita_velocity_from_pressure_gradient
     - ``pytta.velocity_from_pressure_gradient``
     - #105

Room acoustics
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_roomacoustics
     - ``pytta.RoomAnalysis``
     - in PyTTa
   * - ita_roomacoustics_lateral, ita_roomacoustics_IACC
     - ``pytta.iso3382.lateral_fraction``, ``pytta.iso3382.iacc``
     - #66
   * - ita_roomacoustics_late_lateral_sound_level
     - ``pytta.iso3382.late_lateral_level``
     - #86
   * - ita_speech_transmission_index
     - ``pytta.iec60268.sti``
     - #67
   * - ita_sabine, ita_reverberation_chamber_absorption_coefficient
     - ``pytta.iso354.sabine``, ``pytta.iso354.absorption_coefficient``
     - #78
   * - ita_ISO_limits_absorption
     - ``pytta.iso354.max_room_absorption``
     - #114
   * - ita_roomacoustics_imagesources
     - ``pytta.rooms.image_source_ir``
     - #101
   * - ita_roomacoustics_analytic_FRF
     - ``pytta.rooms.rectangular_room_frf``, ``pytta.rooms.rectangular_room_ir``
     - #100
   * - ita_din18041_reverberation_times
     - ``pytta.rooms.din18041``
     - #99
   * - ita_constants, ita_convert_RT_alpha_R_Z
     - ``pytta.rooms.air_constants``, ``pytta.rooms.convert_room_quantity``
     - #99

Building acoustics and materials
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_soundInsulationIndexAirborne, ita_soundInsulationIndexImpact
     - ``pytta.iso717.airborne``, ``pytta.iso717.impact``
     - #65, #84
   * - ita_sound_power
     - ``pytta.iso3741.sound_power``
     - #85
   * - ita_kundt_calc, ita_impedance_tube_calculation
     - ``pytta.iso10534.transfer_function``, ``pytta.iso10534.absorption``
     - #77
   * - ita_impedance_tube_transmission
     - ``pytta.e2611.transfer_matrix``, ``pytta.e2611.transmission_loss``
     - #115
   * - Impedanz, schicht, lochplatte
     - ``pytta.absorbers.surface_impedance``, ``pytta.absorbers.mpp_impedance``
     - #102
   * - ita_analytic_green_function, ita_analytic_plane_wave_on_sphere
     - ``pytta.analytic.green``, ``pytta.analytic.plane_wave_on_sphere``
     - #103

Levels and psychoacoustics
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_sound_level_meter_offline
     - ``pytta.iec61672.leq``, ``pytta.iec61672.time_weighted_level``
     - #72
   * - ita_loudness, ita_psychoacoustic_loudness
     - ``pytta.psychoacoustics.loudness_zwicker``
     - #79
   * - ita_sharpness, ita_psychoacoustic_sharpness
     - ``pytta.psychoacoustics.sharpness_din45692``
     - #79
   * - ita_roughness_daniel, ita_psychoacoustic_roughness
     - ``pytta.psychoacoustics.roughness_daniel1997``
     - #79
   * - ita_psychoacoustic_fluctuationStrength
     - ``pytta.psychoacoustics.fluctuation_strength_osses2016``
     - #92
   * - ita_tonality
     - ``pytta.din45681.tonal_audibility``
     - #88
   * - ita_sone2phon, ita_freq2cb
     - ``pytta.psychoacoustics.loudness_level``, ``pytta.psychoacoustics.critical_band``
     - #93
   * - ita_loudness_adjustment, ita_loudness_compare_rir
     - ``pytta.psychoacoustics.loudness_match``, ``pytta.psychoacoustics.compare_loudness``
     - #94

Loudspeakers and lumped elements
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - ita_thiele_small, ita_linear_loudspeaker
     - ``pytta.loudspeaker.thiele_small``, ``pytta.loudspeaker.impedance_model``
     - #82
   * - ita_thiele_small_fit, ita_thiele_small_fit_weighted
     - ``pytta.loudspeaker.fit_with_velocity``
     - #116
   * - ita_thielesmall2velocity, ita_loudspeakertools_maxSPL
     - ``pytta.loudspeaker.response``, ``pytta.loudspeaker.max_spl``
     - #116
   * - ita_radiation_impedance, ita_calculate_horn_geometry
     - ``pytta.loudspeaker.radiation_impedance``, ``pytta.loudspeaker.horn_profile``
     - #116
   * - ita_inductance_fit, ita_curvefit_inductance
     - ``pytta.loudspeaker.fit_voice_coil``
     - #116
   * - itaFourpole, ita_make_fourpole
     - ``pytta.fourpole.element``, ``pytta.fourpole.chain``
     - #106
   * - ita_integrate, ita_differentiate
     - ``pytta.vibration.integrate``, ``pytta.vibration.differentiate``
     - #76

Simulation
----------

.. list-table::
   :header-rows: 1
   :widths: 40 45 15

   * - ITA-Toolbox
     - PyTTa
     - Status
   * - FluidFEM application
     - ``pytta.fem.modes``, ``pytta.fem.harmonic``
     - #122
   * - (no BEM in the ITA-Toolbox)
     - ``pytta.bem.surface_pressure``, ``pytta.bem.field``
     - #123
   * - RAVEN (ray tracing part)
     - ``pytta.rays.trace``
     - #124
   * - itaMesh, Comsol meshes
     - ``pytta.mesh.distmesh``, ``pytta.mesh.read_mesh``
     - #125

What is not ported
------------------

Besides the functions above, the ITA-Toolbox has:

* functions that one NumPy or SciPy call replaces, such as ``ita_abs``,
  ``ita_conj``, ``ita_sum`` or ``ita_struve1`` (``scipy.special.struve``);
* the infrastructure of MATLAB and of the toolbox itself: graphical
  interfaces, menu callbacks, preferences, path handling and plot tools;
* drivers and setups of the hardware of the ITA laboratory, the playback and
  recording through playrec and MIDI, for which PyTTa uses sounddevice, and
  the interface to the VA auralization server;
* large applications that are not ported yet: spherical harmonics beyond
  Ambisonics, binaural technology (HRTF, headphone equalization, crosstalk
  cancellation), beamforming, transfer path analysis, listening tests and
  sound field classification.
