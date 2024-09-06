import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyroomacoustics as pra

from aspsim.simulator import SimulatorSetup
import aspsim.room.roomimpulseresponse as rir
import aspsim.room.generatepoints as gp

def _setup_ism(fig_folder, samplerate):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = samplerate
    setup.sim_info.tot_samples =  samplerate
    setup.sim_info.sim_chunk_size = 2*samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency =  samplerate
    setup.sim_info.save_source_contributions = False
    setup.sim_info.randomized_ism = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [4, 3, 3]
    setup.sim_info.room_center = [-1, 0, 0]
    setup.sim_info.rt60 = 0.25
    setup.sim_info.max_room_ir_length = samplerate // 2
    return setup


def test_directional_microphones_are_correct_with_regards_to_reciprocity():
    """
    When the directional microphone is pointed towards the source or one of the two reflections, the impulse response for that particular reflection should be identical to the impulse response of the omni-directional microphone.
    
    change src_azimuth in the calculation of mic_dir to ref1_azimuth or ref2_azimuth to test the reflections. 
    """
    pos_src = np.array([[1, 1, 0]])
    pos_mic = np.array([[-2.5, -3, 0]])
    room_size = np.array([150, 8, 150])
    room_center = np.array([0, 0, 0])
    ir_len = 1024
    sr = 4000
    e_absorbtion = 0.5
    max_order = 1
    min_dly = 0

    src_angle = pos_src - pos_mic
    src_angle = src_angle / np.linalg.norm(src_angle, axis=-1, keepdims=True)
    src_azimuth = np.arctan2(src_angle[0,1], src_angle[0,0])


    wall_bottom = room_center[1] - room_size[1] / 2
    wall_top = room_center[1] + room_size[1] / 2
    y1 = pos_mic[0,1] - wall_bottom
    y2 = pos_src[0,1] - wall_bottom
    ref1_azimuth = -np.arctan2((y1 + y2), (np.abs(pos_src[0,0] - pos_mic[0,0]))) #correct for bottom walls

    y1 = wall_top - pos_mic[0,1]
    y2 = wall_top - pos_src[0,1]
    ref2_azimuth = np.arctan2((y1 + y2), (np.abs(pos_src[0,0] - pos_mic[0,0]))) #correct for top walls

    mic_dir = gp.spherical2cart(np.ones(1,), np.array([[src_azimuth, np.pi/2]]))

    ir_omni = rir.ir_room_image_source_3d(pos_src,
    pos_mic,
    room_size,
    room_center,
    ir_len,
    sr,
    e_absorbtion,
    max_order,
    min_dly,
    dir_type_mic = None,
    dir_dir_mic = None,
    randomized_ism = False,
    calculate_metadata=False,
    verbose=False)

    ir_dir = rir.ir_room_image_source_3d(pos_src,
    pos_mic,
    room_size,
    room_center,
    ir_len,
    sr,
    e_absorbtion,
    max_order,
    min_dly,
    dir_type_mic = ["cardioid"],
    dir_dir_mic = mic_dir,
    randomized_ism = False,
    calculate_metadata=False,
    verbose=False)


    fig, ax = plt.subplots(1, 1)
    corner = [c - sz/2 for c, sz in zip(room_center[:2], room_size[:2])]
    width = room_size[0]
    height = room_size[1]
    ax.add_patch(patches.Rectangle(corner, width, height, edgecolor="k", facecolor="none", linewidth=2, alpha=1))
    
    ax.plot(pos_mic[:,0], pos_mic[:,1], "x", label="Mic")
    ax.plot(pos_src[:,0], pos_src[:,1], "o", label="Src")
    ax.plot([pos_mic[0,0], pos_mic[0,0]+3*mic_dir[0,0]], [pos_mic[0,1], pos_mic[0,1]+3*mic_dir[0,1]], label="Mic dir", linestyle="--")  

    ax.axis("equal")
    margin = 0.5
    ax.set_xlim([np.min((pos_mic[:,0], pos_src[:,0])) -margin, np.max((pos_mic[:,0], pos_src[:,0])) -margin])
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(ir_omni[0,0,:], label="Omni")
    ax.plot(ir_dir[0,0,:], label="Dir")
    ax.legend()
    plt.show()


def test_cardioid_microphone_facing_plane_wave_gives_same_response_as_omni_microphone():
    rng = np.random.default_rng(10)
    sr = 1000

    pos_mic = np.zeros((1,3))

    pw_dir = np.array([[1, 1, 1]])
    pw_dir = pw_dir / np.linalg.norm(pw_dir) # incoming from that direction, not propagation direction
    mic_dir = pw_dir
    
    src_distance = 500
    pos_src = pw_dir * src_distance

    setup = _get_default_simulator_setup(sr)
    prop_delay = np.linalg.norm(pos_src) / setup.sim_info.c
    rir_len = int(2 * prop_delay * sr)

    setup.sim_info.room_size = [3*src_distance, 3*src_distance, 3*src_distance]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 =  0
    setup.sim_info.max_room_ir_length = rir_len
    setup.sim_info.sim_buffer = rir_len
    setup.sim_info.extra_delay = 100

    setup.add_mics("omni", pos_mic)
    setup.add_mics("cardioid", pos_mic, directivity_type=["cardioid"], directivity_dir=mic_dir)
    setup.add_controllable_source("src", pos_src)
    sim = setup.create_simulator()

    fig, axes = plt.subplots(2,1)
    axes[0].plot(sim.arrays.paths["src"]["omni"][0,0,:])
    axes[0].plot(sim.arrays.paths["src"]["cardioid"][0,0,:])
    axes[1].plot(sim.arrays.paths["src"]["cardioid"][0,0,:] - sim.arrays.paths["src"]["omni"][0,0,:])
    plt.show()

    mse = np.mean(np.abs(sim.arrays.paths["src"]["omni"][0,0,:] - sim.arrays.paths["src"]["cardioid"][0,0,:])**2) / \
        np.mean(np.abs(sim.arrays.paths["src"]["omni"][0,0,:])**2)
    
    assert mse < 1e-10

    #num_freqs = setup.sim_info.max_room_ir_length
    #fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)

    #return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], sim.arrays.paths["src"]["omni"], \
    #        sim.arrays["cardioid"].pos, fpaths["src"]["cardioid"][...,0], freqs, sim.sim_info


def test_cardioid_microphone_response_compared_to_omni_for_plane_wave_pyroomacoustics():
    samplerate = 1000

    #Plane wave incoming from the following direction, which is negative of propagation direction
    pw_azimuth = 0.234 * np.pi # arbitrary angle
    pw_colatitude = np.pi / 2 + 0.4 * np.pi
    pw_dir = np.array([np.cos(pw_azimuth) * np.sin(pw_colatitude), np.sin(pw_azimuth) * np.sin(pw_colatitude), np.cos(pw_colatitude)])
    
    src_distance = 500
    room_size = np.array([3*src_distance, 3*src_distance, 3*src_distance]) # make room big enough
    room_center = room_size / 2
    pos_mic = room_center[:,np.newaxis]
    pos_src = room_center + pw_dir * src_distance # Put source very far away to approximate plane wave

    # Direct microphone towards the plane wave
    dir_obj = pra.directivities.CardioidFamily(
        orientation=pra.directivities.DirectionVector(azimuth=pw_azimuth, colatitude=pw_colatitude, degrees=False),
        pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
    )

    # RIR for cardioid microphone
    room = pra.ShoeBox(room_size, fs=samplerate, max_order=0, use_rand_ism = False)
    room.add_source(pos_src)
    mics = pra.MicrophoneArray(pos_mic, samplerate, directivity=dir_obj)
    room.add_microphone_array(mics)

    room.compute_rir()
    rir_cardioid = room.rir[0][0]

    # RIR for omni microphone
    room = pra.ShoeBox(room_size, fs=samplerate, max_order=0, use_rand_ism = False)
    room.add_source(pos_src)
    mics = pra.MicrophoneArray(pos_mic, samplerate)
    room.add_microphone_array(mics)
    room.compute_rir()
    rir_omni = room.rir[0][0]

    fig, axes= plt.subplots(2, 1)
    axes[0].plot(rir_cardioid, label = "Cardioid")
    axes[0].plot(rir_omni, label = "Omni")
    axes[0].legend()
    axes[1].plot(rir_cardioid - rir_omni, label = "Difference")
    axes[1].legend()
    plt.show()



def test_cardioid_microphone_response_to_plane_wave_of_different_heights_pyroomacoustics():
    samplerate = 1000

    #Plane wave incoming from the following direction, which is negative of propagation direction
    pw_azimuth = 0.234 * np.pi # arbitrary angle
    pw_colatitude = np.pi / 2
    pw_dir = np.array([np.cos(pw_azimuth) * np.sin(pw_colatitude), np.sin(pw_azimuth) * np.sin(pw_colatitude), np.cos(pw_colatitude)])
    
    src_distance = 500
    room_size = np.array([3*src_distance, 3*src_distance, 3*src_distance]) # make room big enough
    room_center = room_size / 2
    pos_mic = room_center[:,np.newaxis]
    pos_src = room_center + pw_dir * src_distance # Put source very far away to approximate plane wave

    

    # RIR for cardioid microphone
    # Direct microphone towards the plane wave
    dir_obj = pra.directivities.CardioidFamily(
        orientation=pra.directivities.DirectionVector(azimuth=pw_azimuth, colatitude=pw_colatitude, degrees=False), pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
    )
    room = pra.ShoeBox(room_size, fs=samplerate, max_order=0, use_rand_ism = False)
    room.add_source(pos_src)
    mics = pra.MicrophoneArray(pos_mic, samplerate, directivity=dir_obj)
    room.add_microphone_array(mics)

    room.compute_rir()
    rir_cardioid = room.rir[0][0]

    # RIR for plane wave with slightly rotated mic
    # Direct microphone not exactly towards the plane wave
    dir_obj = pra.directivities.CardioidFamily(
        orientation=pra.directivities.DirectionVector(azimuth=pw_azimuth, colatitude=pw_colatitude + 0.3*np.pi, degrees=False), pattern_enum=pra.directivities.DirectivityPattern.CARDIOID,
    )
    room = pra.ShoeBox(room_size, fs=samplerate, max_order=0, use_rand_ism = False)
    room.add_source(pos_src)
    mics = pra.MicrophoneArray(pos_mic, samplerate, directivity=dir_obj)
    room.add_microphone_array(mics)
    room.compute_rir()
    rir_omni = room.rir[0][0]

    fig, axes= plt.subplots(2, 1)
    axes[0].plot(rir_cardioid, label = "Cardioid")
    axes[0].plot(rir_omni, label = "Omni")
    axes[0].legend()
    axes[1].plot(rir_cardioid - rir_omni, label = "Difference")
    axes[1].legend()
    plt.show()



def _generate_plane_wave_with_ism(sr, pw_dir, pos_mic=None, pos_dir=None, mic_dir=None):
    rng = np.random.default_rng(10)
    side_len = 0.5
    num_mic = 40

    assert np.allclose(np.linalg.norm(pw_dir), 1)

    #pos_mic = np.zeros((num_mic, 3))
    if pos_mic is None:
        pos_mic = rng.uniform(low=-side_len/2, high=side_len/2, size=(num_mic, 3))

    if pos_dir is None:
        pos_dir = np.zeros((1,3))
        mic_dir = np.array([[1,0,0]])
    
    pos_src = pw_dir * 500 #np.array([[0,200,0]])

    setup = _get_default_simulator_setup(sr)
    prop_delay = np.linalg.norm(pos_src) / setup.sim_info.c
    rir_len = int(2 * prop_delay * sr)

    setup.sim_info.room_size = [2000, 2000, 2000]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 =  0
    setup.sim_info.max_room_ir_length = rir_len
    setup.sim_info.sim_buffer = rir_len
    setup.sim_info.extra_delay = 100

    setup.add_mics("omni", pos_mic)
    setup.add_mics("cardioid", pos_dir, directivity_type=pos_dir.shape[0]*["cardioid"], directivity_dir=mic_dir)
    setup.add_controllable_source("src", pos_src)
    sim = setup.create_simulator()

    num_freqs = setup.sim_info.max_room_ir_length
    fpaths, freqs = sim.arrays.get_freq_paths(num_freqs, sr)

    return sim.arrays["omni"].pos, fpaths["src"]["omni"][...,0], sim.arrays.paths["src"]["omni"], \
            sim.arrays["cardioid"].pos, fpaths["src"]["cardioid"][...,0], freqs, sim.sim_info





def _get_default_simulator_setup(sr):
    setup = SimulatorSetup()
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.25
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 60
    setup.sim_info.plot_output = "none"
    return setup