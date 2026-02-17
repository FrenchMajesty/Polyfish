// This test file is disabled because `polyfish::ai::heuristics` module is missing.
// It seems to be a scratchpad for previewing heuristics which are no longer available in that path.

/*
use polyfish::actions::units::{remove_unit, summon_unit};
use polyfish::ai::heuristics;
use polyfish::states::{GameState, TileState, TribeState};
use polyfish::types::{TerrainType, TribeType};

fn setup_basic_state() -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    let tribe_id = 1;
    state.settings.current_player_turn_id = tribe_id;

    let mut tribe = TribeState::default();
    tribe.id = tribe_id;
    tribe.tribe_type = TribeType::Imperius;
    state.tribes.insert(tribe_id, tribe);

    // Fill map with fields
    for i in 0..121 {
        let mut tile = TileState::default();
        tile.terrain_type = TerrainType::Field;
        state.tiles.insert(i, tile);
    }

    state
}

fn evaluate_unit(state: &mut GameState, unit_type: polyfish::UnitType) -> f32 {
    let tribe_id = 1;
    let tile_idx = 0; // Use tile 0 for testing

    // Ensure tile is empty/valid
    if let Some(tile) = state.tiles.get_mut(&tile_idx) {
        tile._unit_owner_id = None;
    }

    // Summon unit for tribe 1
    let _ = summon_unit(state, unit_type, tile_idx, false, false);

    // Get the unit (it should be the last one added)
    let unit_score = if let Some(tribe) = state.tribes.get(&tribe_id) {
        if let Some(unit) = tribe.units.last() {
            heuristics::assess_unit_power(state, unit)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Clean up: Remove the unit
    if let Some(tribe) = state.tribes.get(&tribe_id) {
        let unit_idx = tribe.units.len() - 1;
        let _ = remove_unit(state, tribe_id, unit_idx, None, None);
    }

    unit_score
}

#[test]
fn preview_heuristics() {
    let mut state = setup_basic_state();

    println!(
        "Warrior: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Warrior)
    );
    // ... rest of the test ...
}
*/
