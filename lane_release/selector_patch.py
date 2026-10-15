"""Extend the existing Experimental picker to immutable local CD210 candidates."""
import ast

_OLD_IMPORT = '''from iqpilot.selfdrive.iqmodeld.cd210_centerline_candidate import (
  CANDIDATE_REF as CENTERLINE_CANDIDATE_REF,
  DISPLAY_NAME as CENTERLINE_CANDIDATE_NAME,
  select_candidate as select_centerline_candidate,
  is_selected as is_centerline_selected,
)
'''
_IMPORT = '''from iqpilot.selfdrive.iqmodeld.cd210_centerline_candidates_v2 import (
  candidates as centerline_candidates,
  is_candidate_ref as is_centerline_candidate_ref,
  select_candidate as select_centerline_candidate,
  selected_ref as selected_centerline_ref,
)
'''
_OLD_BRANCH = '''    if selected_ref == CENTERLINE_CANDIDATE_REF:
      if _is_c3x():
        try:
          select_centerline_candidate(ui_state.params, offroad=ui_state.is_offroad())
        except RuntimeError as error:
          gui_app.set_modal_overlay(NoticeModal(text=html.escape(str(error))))
        else:
          self.full_frame_candidate_item.action_item.set_state(False)
          self.current_model_item.action_item.set_value(tr(CENTERLINE_CANDIDATE_NAME))
          self._show_reset_params_dialog()
    elif selected_ref == CANDIDATE_REF:
'''
_BRANCH = '''    if is_centerline_candidate_ref(selected_ref):
      if _is_c3x():
        try:
          selected = select_centerline_candidate(ui_state.params, offroad=ui_state.is_offroad(), ref=selected_ref)
        except RuntimeError as error:
          gui_app.set_modal_overlay(NoticeModal(text=html.escape(str(error))))
        else:
          self.full_frame_candidate_item.action_item.set_state(False)
          self.current_model_item.action_item.set_value(tr(selected['displayName']))
          self._show_reset_params_dialog()
    elif selected_ref == CANDIDATE_REF:
'''
_OLD_ITEMS = '''      groups.insert(1, PickerGroup(tr("Experimental"), [
        PickerItem(CENTERLINE_CANDIDATE_REF, {
          'display_name': tr(CENTERLINE_CANDIDATE_NAME),
          'short_name': tr("CD210, checkpoint step 24"),
        }),
'''
_ITEMS = '''      groups.insert(1, PickerGroup(tr("Experimental"), [
        *[PickerItem(item['ref'], {
          'display_name': tr(item['display_name']),
          'short_name': tr(item['short_name']),
        }) for item in centerline_candidates()],
'''


def _replace(source, before, after):
    if source.count(before) != 1:
        raise ValueError('unexpected model picker source: ' + before.splitlines()[0])
    return source.replace(before, after, 1)


def patch_selector(source):
    marker = 'from iqpilot.selfdrive.iqmodeld.cd210_centerline_candidates_v2 import ('
    if marker in source:
        # Do not silently accept a partial or externally modified installation.
        for fragment in (_IMPORT, _BRANCH, _ITEMS,
                         '    active_ref = (selected_centerline_ref(ui_state.params)\n',
                         '    if not target or is_centerline_candidate_ref(target.ref):\n'):
            if source.count(fragment) != 1:
                raise ValueError('incomplete multi-candidate model picker')
        ast.parse(source)
        return source
    original = source
    if _OLD_IMPORT in source:
        source = _replace(source, _OLD_IMPORT, _IMPORT)
        source = _replace(source, _OLD_BRANCH, _BRANCH)
        source = _replace(source, _OLD_ITEMS, _ITEMS)
        source = _replace(source,
            '    active_ref = (CENTERLINE_CANDIDATE_REF if is_centerline_selected(ui_state.params)\n',
            '    active_ref = (selected_centerline_ref(ui_state.params)\n')
        source = _replace(source,
            '                  else CANDIDATE_REF if is_candidate_selected(ui_state.params)\n',
            '                  or (CANDIDATE_REF if is_candidate_selected(ui_state.params)\n')
        source = _replace(source,
            '                  if self._has_active_bundle_param() and self.model_manager.activeBundle else "Default")\n',
            '                  if self._has_active_bundle_param() and self.model_manager.activeBundle else "Default"))\n')
        source = _replace(source, '    if not target or target.ref == CENTERLINE_CANDIDATE_REF:\n',
                          '    if not target or is_centerline_candidate_ref(target.ref):\n')
    else:
        anchor = 'from iqpilot.selfdrive.iqmodeld.full_frame_candidate import ('
        source = _replace(source, anchor, _IMPORT + anchor)
        source = _replace(source, '    if selected_ref == CANDIDATE_REF:\n', _BRANCH)
        source = _replace(source, '      groups.insert(1, PickerGroup(tr("Development"), [\n', _ITEMS)
        source = _replace(source,
            '    active_ref = (CANDIDATE_REF if is_candidate_selected(ui_state.params)\n',
            '    active_ref = (selected_centerline_ref(ui_state.params)\n'
            '                  or (CANDIDATE_REF if is_candidate_selected(ui_state.params)\n')
        source = _replace(source,
            '                  if self._has_active_bundle_param() and self.model_manager.activeBundle else "Default")\n',
            '                  if self._has_active_bundle_param() and self.model_manager.activeBundle else "Default"))\n')
        source = _replace(source, '    if not target:\n      return None\n',
                          '    if not target or is_centerline_candidate_ref(target.ref):\n      return None\n')
    if source.count('toggle_item') != original.count('toggle_item'):
        raise ValueError('candidate picker must not add toggles')
    ast.parse(source)
    return source
