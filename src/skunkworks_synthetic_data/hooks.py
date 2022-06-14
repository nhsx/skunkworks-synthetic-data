"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal
from kedro.pipeline.node import Node
from kedro.pipeline import Pipeline
from kedro.extras.datasets.matplotlib import MatplotlibWriter


class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


class FormatTestOutputHooks:
    def open_html_file(self):
        html_file = open("data/08_reporting/data_checks.html", "w")
        self.html_file = html_file

    def report_section_title(self, title: str) -> str:
        return f"""<hr style="width:50%;text-align:left;margin-left:0">
        <h2>{title}</h2>"""

    def display_text(self, text: str) -> str:
        return f"<h4>{text}</h4>"

    def display_image(self, image_path: str) -> str:
        import base64

        with open(
            image_path,
            "rb",
        ) as img_file:
            image_as_base_64 = base64.b64encode(img_file.read()).decode("utf-8")
            return self.display_base_64_image(image_as_base_64)

    def display_base_64_image(self, image_as_base_64):
        return f"""<img src="data:image/png;base64, {image_as_base_64}" alt="plot" max-width:50% max-height:50%>"""

    @hook_impl
    def before_pipeline_run(self, pipeline: Pipeline) -> None:
        self.open_html_file()
        self.html_file.write(
            """
        <!DOCTYPE html>
        <html>
        <head>
        <style type="text/css">
        body {font-family: Arial, sans-serif;}
        table {
        font-family: sans-serif;
        width: 80%;}

        td, th {
        padding: 8px;
        border: 1px solid #ddd;}

        tr:nth-child(even){background-color: #f2f2f2;}

        tr:hover {background-color: #ddd;}

        th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: center;}
        </style>

        <h1>Comparison metrics between original data and synthetic data</h1>

                    </head>
                    <body>
        """
        )

    @hook_impl
    def after_node_run(self, node: Node, outputs) -> None:
        if "data_evalulation_test" in node.tags:
            self.html_file.write(self.report_section_title(node.name))

            for output in outputs.keys():
                if type(outputs[output]) == pd.DataFrame:
                    self.html_file.write(outputs[output].to_html())

                elif ("figure" in str(output)) & ("images_as_base_64" in node.tags):
                    self.html_file.write(
                        self.display_base_64_image(outputs[f"{node.name}_figure"])
                    )

                else:
                    self.html_file.write(
                        self.display_text(f"""{output}: {outputs[output]}""")
                    )
