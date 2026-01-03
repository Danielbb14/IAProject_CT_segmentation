import io
import gzip
import requests
import copy
import threading
import time
import importlib.util
import numpy as np
from pathlib import Path

import slicer
import qt
import vtk
from qt import QApplication, QPalette
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from PythonQt.QtGui import QMessageBox

###############################################################################
# Decorators and utility functions
###############################################################################

DEBUG_MODE = False

def debug_print(*args):
    if DEBUG_MODE:
        print(*args)

def ensure_synched(func):
    def inner(self, *args, **kwargs):
        failed_to_sync = False
        if self.image_changed():
            debug_print("Image changed. Calling upload_image_to_server()")
            result = self.upload_image_to_server()
            failed_to_sync = result is None

        if not failed_to_sync and self.selected_segment_changed():
            debug_print("Segment changed. Calling upload_segment_to_server()")
            self.remove_all_but_last_prompt()
            result = self.upload_segment_to_server()
            failed_to_sync = result is None

        if not failed_to_sync:
            return func(self, *args, **kwargs)
    return inner

###############################################################################
# SlicerNNInteractive (Branded as CT Image)
###############################################################################

class SlicerNNInteractive(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        # --- REVISION: Title set to CT Image ---
        self.parent.title = _("CT Image")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Coen de Vente", "Kiran Vaidhya Venkadesh", "Your Name"]
        self.parent.helpText = """
            CT Segmentation plugin using SAM (Segment Anything Model).
            """
        self.parent.acknowledgementText = """Based on SlicerNNInteractive."""

###############################################################################
# SlicerNNInteractiveWidget
###############################################################################

class SlicerNNInteractiveWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.install_dependencies()

        ui_widget = slicer.util.loadUI(self.resourcePath("UI/SlicerNNInteractive.ui"))
        self.layout.addWidget(ui_widget)
        self.ui = slicer.util.childWidgetVariables(ui_widget)

        # --- REVISION: DEEP SEARCH AND REPLACE ---
        try:
            for var_name, widget in self.ui.__dict__.items():

                # 1. Rename standard widgets (Labels, Buttons, CollapsibleButtons)
                # --- CHANGE: Updated target text to "CT segmentation" ---
                if hasattr(widget, "text"):
                    if "nninteractive" in widget.text.lower():
                        widget.text = "CT segmentation"
                    # Also catch the generic "Prompts" if needed
                    if widget.text == "Prompts":
                         widget.text = "CT segmentation"

                # 2. Rename TABS
                if isinstance(widget, qt.QTabWidget):
                    for i in range(widget.count):
                        tab_text = widget.tabText(i).lower()
                        if "nninteractive" in tab_text:
                            # Rename the tab to "Segmentation" to match your target image
                            widget.setTabText(i, "Segmentation")

        except Exception as e:
            debug_print(f"Could not auto-rename widgets: {e}")

        # --- REVISION: Hide unused tools (Lasso/Scribble) ---
        if hasattr(self.ui, "pbInteractionLasso"): self.ui.pbInteractionLasso.setVisible(False)
        if hasattr(self.ui, "pbInteractionScribble"): self.ui.pbInteractionScribble.setVisible(False)
        if hasattr(self.ui, "pbInteractionLassoCancel"): self.ui.pbInteractionLassoCancel.setVisible(False)

        # --- REVISION: Rename Point button to "SAM Point" ---
        if hasattr(self.ui, "pbInteractionPoint"):
            self.ui.pbInteractionPoint.setText("SAM Point")

        # Setup Editor
        self.ui.editor_widget.setMaximumNumberOfUndoStates(10)
        self.ui.editor_widget.setMRMLScene(slicer.mrmlScene)
        segment_editor_singleton_tag = "SegmentEditor"
        self.segment_editor_node = slicer.mrmlScene.GetSingletonNode(segment_editor_singleton_tag, "vtkMRMLSegmentEditorNode")
        if self.segment_editor_node is None:
            self.segment_editor_node = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            self.segment_editor_node.UnRegister(None)
            self.segment_editor_node.SetSingletonTag(segment_editor_singleton_tag)
            self.segment_editor_node = slicer.mrmlScene.AddNode(self.segment_editor_node)
        self.ui.editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.ui.editor_widget.setSegmentationNode(self.get_segmentation_node())

        self.selected_style = "background-color: #3498db; color: white"
        self.unselected_style = ""

        # --- REVISION: Simplified Prompt Types (Only SAM Point and BBox) ---
        self.prompt_types = {
            "point": {
                "node_class": "vtkMRMLMarkupsFiducialNode",
                "node": None,
                "name": "SamPointPrompt",
                "display_node_markup_function": self.display_node_markup_point,
                "on_placed_function": self.on_point_placed,
                "button": self.ui.pbInteractionPoint,
                "button_text": "SAM Point",
                "button_icon_filename": "point_icon.svg",
            },
            "bbox": {
                "node_class": "vtkMRMLMarkupsROINode",
                "node": None,
                "name": "BBoxPrompt",
                "display_node_markup_function": self.display_node_markup_bbox,
                "on_placed_function": self.on_bbox_placed,
                "button": self.ui.pbInteractionBBox,
                "button_text": self.ui.pbInteractionBBox.text,
                "button_icon_filename": "bbox_icon.svg",
            }
        }

        self.setup_shortcuts()
        self.all_prompt_buttons = {}
        self.setup_prompts()
        self.init_ui_functionality()

        _ = self.get_current_segment_id()
        self.previous_states = {}

    def init_ui_functionality(self):
        self.ui.uploadProgressGroup.setVisible(False)
        savedServer = slicer.util.settingsValue("SlicerNNInteractive/server", "")
        self.ui.Server.text = savedServer
        self.server = savedServer.rstrip("/")
        self.ui.Server.editingFinished.connect(self.update_server)

        # Set initial prompt type
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)

        # Top buttons
        self.ui.pbResetSegment.clicked.connect(self.clear_current_segment)
        self.ui.pbNextSegment.clicked.connect(self.make_new_segment)

        # Prompt Type buttons
        self.ui.pbPromptTypePositive.clicked.connect(self.on_prompt_type_positive_clicked)
        self.ui.pbPromptTypeNegative.clicked.connect(self.on_prompt_type_negative_clicked)

        self.addObserver(slicer.app.applicationLogic().GetInteractionNode(),
            slicer.vtkMRMLInteractionNode.InteractionModeChangedEvent, self.on_interaction_node_modified)

    def setup_shortcuts(self):
        shortcuts = {
            "o": self.ui.pbInteractionPoint.click,
            "b": self.ui.pbInteractionBBox.click,
            "e": self.make_new_segment,
            "r": self.clear_current_segment,
            "t": self.toggle_prompt_type,
        }
        self.shortcut_items = {}
        for key, event in shortcuts.items():
            shortcut = qt.QShortcut(qt.QKeySequence(key), slicer.util.mainWindow())
            shortcut.activated.connect(event)
            self.shortcut_items[key] = shortcut

    def remove_shortcut_items(self):
        if hasattr(self, "shortcut_items"):
            for _, shortcut in self.shortcut_items.items():
                shortcut.setParent(None)
                shortcut.deleteLater()

    def install_dependencies(self):
        dependencies = {
            "requests_toolbelt": "requests_toolbelt",
            "skimage": "scikit-image",
        }
        for dependency in dependencies:
            if self.check_dependency_installed(dependency, dependencies[dependency]):
                continue
            self.run_with_progress_bar(self.pip_install_wrapper, (dependencies[dependency],), "Installing dependencies: %s" % dependency)

    def check_dependency_installed(self, import_name, module_name_and_version):
        if "==" in module_name_and_version:
            module_name, module_version = module_name_and_version.split("==")
        else:
            module_name = module_name_and_version
            module_version = None
        spec = importlib.util.find_spec(import_name)
        if spec is None: return False
        if module_version is not None:
            import importlib.metadata as metadata
            try:
                version = metadata.version(module_name)
                if version != module_version: return False
            except metadata.PackageNotFoundError: pass
        return True

    def pip_install_wrapper(self, command, event):
        slicer.util.pip_install(command)
        event.set()

    def run_with_progress_bar(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 100
        self.progressbar.setLabelText(title)
        parallel_event = threading.Event()
        dep_thread = threading.Thread(target=target, args=(*args, parallel_event,))
        dep_thread.start()
        while not parallel_event.is_set():
            slicer.app.processEvents()
        dep_thread.join()
        self.progressbar.close()

    def cleanup(self):
        self.removeObservers()
        self.remove_shortcut_items()

    def __del__(self):
        self.remove_shortcut_items()

    ###############################################################################
    # Prompt and markup setup functions
    ###############################################################################

    def setup_prompts(self, skip_if_exists=False):
        if not skip_if_exists:
            self.remove_prompt_nodes()

        for prompt_name, prompt_type in self.prompt_types.items():
            if skip_if_exists and slicer.mrmlScene.GetFirstNodeByName(prompt_type["name"]):
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(prompt_type["node_class"])
            node.SetName(prompt_type["name"])
            node.CreateDefaultDisplayNodes()

            display_node = node.GetDisplayNode()
            prompt_type["display_node_markup_function"](display_node)

            prompt_type["button"].setStyleSheet(f"QPushButton {{ {self.unselected_style} }} QPushButton:checked {{ {self.selected_style} }}")

            self.prev_caller = None
            if prompt_type["on_placed_function"] is not None:
                node.AddObserver(slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent, prompt_type["on_placed_function"])

            prompt_type["node"] = node
            prompt_type["button"].clicked.connect(lambda checked, prompt_name=prompt_name: self.on_place_button_clicked(checked, prompt_name))
            self.all_prompt_buttons[prompt_name] = prompt_type["button"]

            light_dark_mode = self.is_ui_dark_or_light_mode()
            icon_path = self.resourcePath(f"Icons/prompts/{light_dark_mode}/{prompt_type['button_icon_filename']}")
            if Path(icon_path).exists():
                prompt_type["button"].setIcon(qt.QIcon(icon_path))

        interaction_node = slicer.app.applicationLogic().GetInteractionNode()
        interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)

    def is_ui_dark_or_light_mode(self):
        current_style = slicer.app.settings().value("Styles/Style")
        if current_style == "Dark Slicer": return "dark"
        elif current_style == "Light Slicer": return "light"
        elif current_style == "Slicer":
            app_palette = QApplication.instance().palette()
            if app_palette.color(QPalette.Active, QPalette.Window).lightness() < 128: return "dark"
        return "light"

    def remove_prompt_nodes(self):
        def _remove(node_name):
            existing_nodes = slicer.mrmlScene.GetNodesByName(node_name)
            if existing_nodes:
                for i in range(existing_nodes.GetNumberOfItems()):
                    slicer.mrmlScene.RemoveNode(existing_nodes.GetItemAsObject(i))

        for prompt_type in self.prompt_types.values():
            _remove(prompt_type["name"])

    def on_interaction_node_modified(self, caller, event):
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        selectionNode = slicer.app.applicationLogic().GetSelectionNode()
        for prompt_type in self.prompt_types.values():
            if interactionNode.GetCurrentInteractionMode() != slicer.vtkMRMLInteractionNode.Place:
                prompt_type["button"].setChecked(False)
            else:
                placingThisNode = (selectionNode.GetActivePlaceNodeID() == prompt_type["node"].GetID())
                prompt_type["button"].setChecked(placingThisNode)

    def remove_all_but_last_prompt(self):
        last_modified_node = None
        all_nodes = []
        for prompt_type in self.prompt_types.values():
            existing_nodes = slicer.mrmlScene.GetNodesByName(prompt_type["name"])
            if existing_nodes:
                for i in range(existing_nodes.GetNumberOfItems()):
                    node = existing_nodes.GetItemAsObject(i)
                    all_nodes.append(node)
                    if last_modified_node is None or node.GetMTime() > last_modified_node.GetMTime():
                        last_modified_node = node

        for node in all_nodes:
            n = node.GetNumberOfControlPoints()
            if node == last_modified_node: n -= 1
            for i in range(n): node.RemoveNthControlPoint(0)

    def on_place_button_clicked(self, checked, prompt_name):
        self.setup_prompts(skip_if_exists=True)
        interactionNode = slicer.app.applicationLogic().GetInteractionNode()
        if checked:
            selectionNode = slicer.app.applicationLogic().GetSelectionNode()
            selectionNode.SetReferenceActivePlaceNodeClassName(self.prompt_types[prompt_name]["node_class"])
            selectionNode.SetActivePlaceNodeID(self.prompt_types[prompt_name]["node"].GetID())
            interactionNode.SetPlaceModePersistence(1)
            interactionNode.SetCurrentInteractionMode(interactionNode.Place)
        else:
            interactionNode.SetCurrentInteractionMode(interactionNode.ViewTransform)

    def display_node_markup_point(self, display_node):
        display_node.SetTextScale(0)
        display_node.SetGlyphScale(0.75)
        display_node.SetColor(0.0, 0.0, 1.0)
        display_node.SetSelectedColor(0.0, 0.0, 1.0)
        display_node.SetActiveColor(0.0, 0.0, 1.0)
        display_node.SetOpacity(1.0)
        display_node.SetSliceProjection(False)

    def display_node_markup_bbox(self, display_node):
        display_node.SetFillOpacity(0)
        display_node.SetOutlineOpacity(0.5)
        display_node.SetSelectedColor(0, 0, 1)
        display_node.SetColor(0, 0, 1)
        display_node.SetActiveColor(0, 0, 1)
        display_node.SetInteractionHandleScale(1)
        display_node.SetGlyphScale(0)
        display_node.SetHandlesInteractive(False)
        display_node.SetTextScale(0)

    ###############################################################################
    # Event handlers for prompts (USING SAM ENDPOINTS)
    ###############################################################################

    def on_point_placed(self, caller, event):
        xyz = self.xyz_from_caller(caller)
        volume_node = self.get_volume_node()
        if volume_node:
            self.point_prompt(xyz=xyz, positive_click=self.is_positive)

    @ensure_synched
    def point_prompt(self, xyz=None, positive_click=False):
        # --- REVISION: Kept SAM Endpoint ---
        url = f"{self.server}/add_fastsam3d_interaction"

        seg_response = self.request_to_server(url, json={"voxel_coord": xyz, "positive_click": positive_click})
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        self.show_segmentation(unpacked_segmentation)

    def on_bbox_placed(self, caller, event):
        xyz = self.xyz_from_caller(caller)
        if self.prev_caller is not None and caller.GetID() == self.prev_caller.GetID():
            roi_node = slicer.mrmlScene.GetNodeByID(caller.GetID())
            current_size = list(roi_node.GetSize())
            drawn_in_axis = np.argwhere(np.array(xyz) == self.prev_bbox_xyz).squeeze()
            current_size[drawn_in_axis] = 0
            roi_node.SetSize(current_size)

            volume_node = self.get_volume_node()
            if volume_node:
                outer_point_two = self.prev_bbox_xyz
                outer_point_one = [xyz[0] * 2 - outer_point_two[0], xyz[1] * 2 - outer_point_two[1], xyz[2] * 2 - outer_point_two[2]]
                self.bbox_prompt(outer_point_one=outer_point_one, outer_point_two=outer_point_two, positive_click=self.is_positive)
                qt.QTimer.singleShot(0, lambda: (self.setup_prompts(), self.ui.pbInteractionBBox.click()))
            self.prev_caller = None
        else:
            self.prev_bbox_xyz = xyz
        self.prev_caller = caller

    @ensure_synched
    def bbox_prompt(self, outer_point_one, outer_point_two, positive_click=False):
        # --- REVISION: Kept SAM Endpoint ---
        url = f"{self.server}/add_fastsam3d_bbox_interaction"
        seg_response = self.request_to_server(url, json={
                "outer_point_one": outer_point_one[::-1],
                "outer_point_two": outer_point_two[::-1],
                "positive_click": positive_click,
            })
        unpacked_segmentation = self.unpack_binary_segmentation(seg_response.content, decompress=False)
        self.show_segmentation(unpacked_segmentation)

    ###############################################################################
    # Segmentation-related functions
    ###############################################################################

    def make_new_segment(self):
        self.ui.pbPromptTypePositive.click()
        segmentation_node = self.get_segmentation_node()
        segment_ids = segmentation_node.GetSegmentation().GetSegmentIDs()
        if len(segment_ids) == 0: new_segment_name = "Segment_1"
        else:
            segment_numbers = [int(seg.split("_")[-1]) for seg in segment_ids if seg.startswith("Segment_") and seg.split("_")[-1].isdigit()]
            next_segment_number = max(segment_numbers) + 1 if segment_numbers else 1
            new_segment_name = f"Segment_{next_segment_number}"
        new_segment_id = segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        self.segment_editor_node.SetSelectedSegmentID(new_segment_id)
        return segmentation_node, new_segment_id

    def clear_current_segment(self):
        self.ui.pbPromptTypePositive.click()
        _, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()
        if selected_segment_id:
            self.show_segmentation(np.zeros(self.get_image_data().shape, dtype=np.uint8))
            self.setup_prompts()
            self.upload_segment_to_server()

    def show_segmentation(self, segmentation_mask):
        self.previous_states["segment_data"] = segmentation_mask
        segmentationNode, selectedSegmentID = self.get_selected_segmentation_node_and_segment_id()
        was_3d_shown = segmentationNode.GetSegmentation().ContainsRepresentation(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
        with slicer.util.RenderBlocker():
            self.ui.editor_widget.saveStateForUndo()
            slicer.util.updateSegmentBinaryLabelmapFromArray(segmentation_mask, segmentationNode, selectedSegmentID, self.get_volume_node())
            if was_3d_shown: segmentationNode.CreateClosedSurfaceRepresentation()
        segment = segmentationNode.GetSegmentation().GetSegment(selectedSegmentID)
        if slicer.vtkSlicerSegmentationsModuleLogic.GetSegmentStatus(segment) == slicer.vtkSlicerSegmentationsModuleLogic.NotStarted:
            slicer.vtkSlicerSegmentationsModuleLogic.SetSegmentStatus(segment, slicer.vtkSlicerSegmentationsModuleLogic.InProgress)
        segmentationNode.Modified()
        if segmentation_mask.sum() > 0: segmentationNode.GetSegmentation().CollapseBinaryLabelmaps()
        del segmentation_mask

    def get_segmentation_node(self):
        segmentation_node = self.ui.editor_widget.segmentationNode()
        if segmentation_node: return segmentation_node
        segmentation_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        for n in segmentation_nodes:
            segmentation_node = n; break
        if not segmentation_node:
            segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.ui.editor_widget.setSegmentationNode(segmentation_node)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.get_volume_node())
        return segmentation_node

    def get_selected_segmentation_node_and_segment_id(self):
        segmentation_node = self.get_segmentation_node()
        selected_segment_id = self.get_current_segment_id()
        if not selected_segment_id: return self.make_new_segment()
        return segmentation_node, selected_segment_id

    def get_current_segment_id(self):
        return self.ui.editor_widget.mrmlSegmentEditorNode().GetSelectedSegmentID()

    def get_segment_data(self):
        segmentation_node, selected_segment_id = self.get_selected_segmentation_node_and_segment_id()
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, selected_segment_id, self.get_volume_node())
        return mask.astype(bool)

    def selected_segment_changed(self):
        segment_data = self.get_segment_data()
        old_segment_data = self.previous_states.get("segment_data", None)
        return old_segment_data is None or not np.array_equal(old_segment_data.astype(bool), segment_data.astype(bool))

    ###############################################################################
    # Server communication
    ###############################################################################

    def update_server(self):
        self.server = self.ui.Server.text.rstrip("/")
        qt.QSettings().setValue("SlicerNNInteractive/server", self.server)

    def request_to_server(self, *args, **kwargs):
        with slicer.util.tryWithErrorDisplay(_("Segmentation failed."), waitCursor=True):
            try:
                response = requests.post(*args, **kwargs)
            except (requests.exceptions.MissingSchema, requests.exceptions.ConnectionError, requests.exceptions.InvalidSchema) as e:
                raise RuntimeError(f"Server error: {e}")
            if response.status_code != 200:
                raise RuntimeError(f"Request failed with status code {response.status_code}")

            if "application/json" in response.headers.get("Content-Type", ""):
                resp_json = response.json()
                if resp_json.get("status") == "error":
                    if "No image uploaded" in resp_json.get("message", ""):
                        self.upload_image_to_server()
                        self.upload_segment_to_server()
                        return self.request_to_server(*args, **kwargs)
                    else:
                        raise RuntimeError(f"Server error: {resp_json.get('message')}")
        return response

    def upload_image_to_server(self):
        try:
            image_data = self.get_image_data()
            if image_data is None: return
            url = f"{self.server}/upload_image"
            buffer = io.BytesIO()
            np.save(buffer, image_data)
            raw_data = buffer.getvalue()
            files = {"file": ("volume.npy", raw_data, "application/octet-stream")}
            from requests_toolbelt import MultipartEncoder
            encoder = MultipartEncoder(fields=files)
            return self.request_to_server(url, data=encoder, headers={"Content-Type": encoder.content_type})
        except Exception as e:
            debug_print(f"Error in upload_image: {e}")

    def upload_segment_to_server(self):
        try:
            segment_data = self.get_segment_data()
            files = self.mask_to_np_upload_file(segment_data)
            url = f"{self.server}/upload_segment"
            return self.request_to_server(url, files=files, headers={"Content-Encoding": "gzip"})
        except Exception as e:
            debug_print(f"Error in upload_segment: {e}")

    ###############################################################################
    # Utility / converters functions
    ###############################################################################

    def get_image_data(self):
        volume_node = self.get_volume_node()
        return slicer.util.arrayFromVolume(volume_node) if volume_node else None

    def get_volume_node(self):
        volumeNode = self.ui.editor_widget.sourceVolumeNode()
        if not volumeNode:
            volumeNodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            if volumeNodes: volumeNode = volumeNodes[-1]
            self.ui.editor_widget.setSourceVolumeNode(volumeNode)
        return volumeNode

    def image_changed(self, do_prev_image_update=True):
        image_data = self.get_image_data()
        if image_data is None: return
        old_image_data = self.previous_states.get("image_data", None)
        image_changed = old_image_data is None or not np.array_equal(old_image_data, image_data)
        if do_prev_image_update: self.previous_states["image_data"] = copy.deepcopy(image_data)
        return image_changed

    def mask_to_np_upload_file(self, mask):
        buffer = io.BytesIO()
        np.save(buffer, mask)
        compressed_data = gzip.compress(buffer.getvalue())
        return {"file": ("volume.npy.gz", compressed_data, "application/octet-stream")}

    def unpack_binary_segmentation(self, binary_data, decompress=False):
        if decompress: binary_data = gzip.decompress(binary_data)
        if self.get_image_data() is None: return
        vol_shape = self.get_image_data().shape
        total_voxels = np.prod(vol_shape)
        unpacked_bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        segmentation_mask = unpacked_bits[:total_voxels].reshape(vol_shape).astype(np.bool_).astype(np.uint8)
        return segmentation_mask

    def ras_to_xyz(self, pos):
        volumeNode = self.get_volume_node()
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, volumeNode.GetParentTransformNode(), transformRasToVolumeRas)
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(pos)
        volumeRasToIjk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volumeRasToIjk)
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(list(point_VolumeRas) + [1.0], point_Ijk)
        return [int(round(c)) for c in point_Ijk[0:3]]

    def xyz_from_caller(self, caller, lock_point=True, point_type="control_point"):
        if point_type == "control_point":
            n = caller.GetNumberOfControlPoints()
            if n < 0: return
            pos = [0, 0, 0]
            caller.GetNthControlPointPosition(n - 1, pos)
            if lock_point: caller.SetNthControlPointLocked(n - 1, True)
            return self.ras_to_xyz(pos)
        return []

    ###############################################################################
    # Prompt type toggle
    ###############################################################################

    @property
    def is_positive(self):
        return self.ui.pbPromptTypePositive.isChecked()

    def on_prompt_type_positive_clicked(self, checked=False):
        self.current_prompt_type_positive = True
        self.ui.pbPromptTypePositive.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypePositive.setChecked(True)
        self.ui.pbPromptTypeNegative.setChecked(False)

    def on_prompt_type_negative_clicked(self, checked=False):
        self.current_prompt_type_positive = False
        self.ui.pbPromptTypePositive.setStyleSheet(self.unselected_style)
        self.ui.pbPromptTypeNegative.setStyleSheet(self.selected_style)
        self.ui.pbPromptTypePositive.setChecked(False)
        self.ui.pbPromptTypeNegative.setChecked(True)

    def toggle_prompt_type(self, checked=False):
        if self.current_prompt_type_positive: self.on_prompt_type_negative_clicked()
        else: self.on_prompt_type_positive_clicked()