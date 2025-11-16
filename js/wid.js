import { app } from "../../scripts/app.js";

export const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = "input_",
  connectionType = "IMAGE"
) => {
  if (!node._isRestoring) {
    node._dynamicInputs = new Set();
  }

  const removeUnusedInputs = () => {
    for (let i = node.inputs.length - 1; i >= 0; i--) {
      const input = node.inputs[i];
      const isDynamic = node._dynamicInputs?.has(input.name);

      if (input.link || (isDynamic && node._isRestoring)) {
        continue;
      }

      if (isDynamic) {
        node.removeInput(i);
        node._dynamicInputs.delete(input.name);
      }
    }
  };

  const prevRestoring = node._isRestoring;
  if (node._isRestoring) {
    node._dynamicInputs = new Set(
      node.inputs
        .filter((input) => input.name.startsWith(connectionPrefix))
        .map((input) => input.name)
    );
  }

  removeUnusedInputs();

  const lastInput = node.inputs[node.inputs.length - 1];
  if (lastInput?.link && !node._isRestoring) {
    const newIndex = node.inputs.length + 1;
    const newName = `${connectionPrefix}${newIndex}`;
    const newInput = node.addInput(newName, connectionType);
    node._dynamicInputs.add(newName);
  }

  let validIndex = 1;
  node.inputs.forEach((input) => {
    if (input.name.startsWith(connectionPrefix)) {
      input.name = `${connectionPrefix}${validIndex++}`;
      input.label = input.name;
    }
  });

  node._isRestoring = prevRestoring;
};

app.registerExtension({
  name: ["ComfyUI-Qwen3-VL.BatchImageLoaderToLocalFiles.image"],

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    switch (nodeData.name) {
      case "BatchImageLoaderToLocalFiles":
        widget(nodeType, nodeData, app);
        break;
    }
  },
});

function widget(nodeType, nodeData, app) {
  const input_name = "image";

  const originalSerialize = nodeType.prototype.serialize;
  nodeType.prototype.serialize = function () {
    const data = originalSerialize?.call(this) || {};
    data._dynamicInputs = Array.from(this._dynamicInputs || []);
    return data;
  };

  const originalConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (data) {
    this._isRestoring = true;
    this._dynamicInputs = new Set(data?._dynamicInputs || []);
    const result = originalConfigure?.call(this, data);

    dynamic_connection(this, -1, false, input_name, "IMAGE");

    this._isRestoring = false;
    return result;
  };

  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const res = onNodeCreated?.apply(this, arguments);
    this.addInput(`${input_name}1`, "IMAGE");
    return res;
  };

  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function (
    type,
    index,
    connected,
    link_info
  ) {
    if (!link_info) return;
    const res = onConnectionsChange?.apply(this, arguments);

    const connectionType = this.inputs[0]?.type || "IMAGE";

    dynamic_connection(this, index, connected, input_name, connectionType);

    return res;
  };
}
