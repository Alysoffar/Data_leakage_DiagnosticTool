"""AST-based checks for preprocessing leakage in Python training code."""

import ast
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from .types import CheckResult

MAX_SOURCE_BYTES = 2_000_000
MAX_AST_NODES = 100_000


class ExtendedPreprocessingLeakVisitor(ast.NodeVisitor):
	"""Find preprocessing fitted outside training or fold-training data."""

	def __init__(self, function_safe_params: dict[str, set[int]] | None = None) -> None:
		self.pipeline_vars: set[str] = set()
		self.dataset_states: dict[str, str] = {}
		self.violations: list[dict[str, Any]] = []
		self._cv_context: list[tuple[str, str]] = []
		self._function_safe_params = function_safe_params or {}

	@property
	def in_cv_loop(self) -> bool:
		return bool(self._cv_context)

	def visit_Assign(self, node: ast.Assign) -> None:
		if isinstance(node.value, ast.Call):
			function_name = self._call_name(node.value)
			if function_name in {"Pipeline", "make_pipeline"}:
				self.pipeline_vars.update(
					name for name in (self._node_name(target) for target in node.targets) if name
				)
			elif function_name == "train_test_split":
				for target in node.targets:
					if isinstance(target, (ast.Tuple, ast.List)):
						output_names = [self._node_name(element) for element in target.elts]
						if output_names:
							self._set_state(output_names[0], "train")
						if len(output_names) > 1:
							self._set_state(output_names[1], "test")
			elif function_name in {"GridSearchCV", "RandomizedSearchCV"}:
				estimator_node = (
					node.value.args[0]
					if node.value.args
					else self._keyword_value(node.value, "estimator")
				)
				if estimator_node is not None and not self._is_pipeline_expression(estimator_node):
					self._add_violation(
						node.value,
						function_name,
						self._node_name(estimator_node),
						f"{function_name} wraps a raw estimator. Put preprocessing inside a Pipeline before hyperparameter search.",
					)

		source_name = self._node_name(node.value)
		if source_name in self.dataset_states:
			state = self.dataset_states[source_name]
			for name in (self._node_name(target) for target in node.targets):
				if name:
					self._set_state(name, state)
		if source_name in self.pipeline_vars:
			self.pipeline_vars.update(
				name for name in (self._node_name(target) for target in node.targets) if name
			)

		if self.in_cv_loop:
			train_index = self._cv_context[-1][1]
			if train_index and self._contains_name(node.value, train_index):
				for name in (self._node_name(target) for target in node.targets):
					if name:
						self._set_state(name, "fold_train")
		self.generic_visit(node)

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		safe_positions = self._function_safe_params.get(node.name, set())
		safe_names = {
			argument.arg
			for position, argument in enumerate(node.args.args)
			if position in safe_positions
		}
		previous = {
			name: self.dataset_states.get(name)
			for name in safe_names
		}
		for name in safe_names:
			self._set_state(name, "train")
		for statement in node.body:
			self.visit(statement)
		for name, state in previous.items():
			if state is None:
				self.dataset_states.pop(name, None)
			else:
				self.dataset_states[name] = state

	def visit_For(self, node: ast.For) -> None:
		context = self._cv_context_from_loop(node)
		if context is None:
			self.generic_visit(node)
			return

		self._cv_context.append(context)
		for statement in node.body:
			self.visit(statement)
		self._cv_context.pop()
		for statement in node.orelse:
			self.visit(statement)

	def visit_Call(self, node: ast.Call) -> None:
		function_name = self._call_name(node)
		if function_name in {"cross_val_score", "cross_validate", "cross_val_predict"}:
			estimator_node = node.args[0] if node.args else self._keyword_value(node, "estimator")
			dataset_node = node.args[1] if len(node.args) >= 2 else self._keyword_value(node, "X")
			estimator = self._node_name(estimator_node) if estimator_node else ""
			dataset = self._node_name(dataset_node) if dataset_node else ""
			if estimator and dataset and estimator_node is not None and not self._is_pipeline_expression(estimator_node):
				self._add_violation(
					node,
					function_name,
					dataset,
					f"{function_name} uses raw estimator '{estimator}'. "
					"Put preprocessing inside a Pipeline before cross-validation.",
				)
		elif function_name in {"fit", "fit_transform", "partial_fit"}:
			dataset_node = node.args[0] if node.args else (
				self._keyword_value(node, "X") or self._keyword_value(node, "x")
			)
			if dataset_node is None:
				self.generic_visit(node)
				return
			dataset = self._node_name(dataset_node)
			if self.in_cv_loop:
				cv_dataset = self._cv_context[-1][0]
				if dataset == cv_dataset:
					self._add_violation(
						node,
						function_name,
						dataset,
						f"{function_name} uses full CV dataset '{dataset}' instead of fold training data.",
					)
			elif self.dataset_states.get(dataset) not in {"train", "fold_train"}:
				self._add_violation(
					node,
					function_name,
					dataset,
					f"{function_name} uses unsafe dataset '{dataset}'. "
					"Fit preprocessing only on training data.",
				)
		self.generic_visit(node)

	def _add_violation(
		self, node: ast.Call, method: str, variable: str, issue: str
	) -> None:
		self.violations.append(
			{
				"line": node.lineno,
				"column": node.col_offset,
				"method": method,
				"variable": variable,
				"issue": issue,
			}
		)

	def _cv_context_from_loop(self, node: ast.For) -> tuple[str, str] | None:
		iterator = node.iter
		if not isinstance(iterator, ast.Call) or self._call_name(iterator) != "split":
			return None
		if not iterator.args or not isinstance(node.target, (ast.Tuple, ast.List)):
			return None
		if not node.target.elts:
			return None
		return self._node_name(iterator.args[0]), self._node_name(node.target.elts[0])

	def _set_state(self, name: str, state: str) -> None:
		if name:
			self.dataset_states[name] = state

	def _is_pipeline_expression(self, node: ast.AST) -> bool:
		if isinstance(node, ast.Name):
			return node.id in self.pipeline_vars
		return self._call_name(node) in {"Pipeline", "make_pipeline"}

	def _keyword_value(self, node: ast.Call, name: str) -> ast.AST | None:
		for keyword in node.keywords:
			if keyword.arg == name:
				return keyword.value
		return None

	def _call_name(self, node: ast.AST) -> str:
		function = node.func if isinstance(node, ast.Call) else node
		if isinstance(function, ast.Name):
			return function.id
		if isinstance(function, ast.Attribute):
			return function.attr
		return ""

	def _node_name(self, node: ast.AST) -> str:
		if isinstance(node, ast.Name):
			return node.id
		if isinstance(node, ast.Attribute):
			return node.attr
		if isinstance(node, ast.Subscript):
			return self._node_name(node.value)
		return ""

	def _contains_name(self, node: ast.AST, name: str) -> bool:
		return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def check_code_leakage_extended(code_str: str, source_name: str | None = None) -> "CheckResult":
	"""Parse Python source and return static preprocessing leakage findings."""
	if len(code_str.encode("utf-8")) > MAX_SOURCE_BYTES:
		return {
			"check": "static_code_leakage",
			"error": f"Source exceeds the {MAX_SOURCE_BYTES}-byte safety limit.",
			"n_flagged": 0,
			"detail": [],
		}
	try:
		tree = ast.parse(code_str, filename=source_name or "<source>")
	except SyntaxError as error:
		return {
			"check": "static_code_leakage",
			"error": f"SyntaxError: {error}",
			"n_flagged": 0,
			"detail": [],
		}
	if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
		return {
			"check": "static_code_leakage",
			"error": f"Source exceeds the {MAX_AST_NODES}-node AST safety limit.",
			"n_flagged": 0,
			"detail": [],
		}

	function_defs: dict[str, ast.FunctionDef] = {
		node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
	}
	split_outputs: set[str] = set()
	for node in ast.walk(tree):
		if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
			continue
		if not isinstance(node.value.func, ast.Name) or node.value.func.id != "train_test_split":
			continue
		for target in node.targets:
			if isinstance(target, (ast.Tuple, ast.List)):
				split_outputs.update(
					element.id for element in target.elts if isinstance(element, ast.Name)
				)
	function_safe_params: dict[str, set[int]] = {name: set() for name in function_defs}
	for node in ast.walk(tree):
		if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
			continue
		definition = function_defs.get(node.func.id)
		if definition is None:
			continue
		for position, argument in enumerate(node.args):
			if isinstance(argument, ast.Name) and argument.id in split_outputs:
				function_safe_params[node.func.id].add(position)

	visitor = ExtendedPreprocessingLeakVisitor(function_safe_params)
	visitor.visit(tree)
	return {
		"check": "static_code_leakage",
		"n_flagged": len(visitor.violations),
		"detail": visitor.violations,
	}


__all__ = ["ExtendedPreprocessingLeakVisitor", "check_code_leakage_extended"]
