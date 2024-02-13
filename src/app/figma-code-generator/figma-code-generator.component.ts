import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { DomSanitizer } from '@angular/platform-browser'
import { Router } from '@angular/router';

@Component({
  selector: 'app-figma-code-generator',
  templateUrl: './figma-code-generator.component.html',
  styleUrl: './figma-code-generator.component.scss',
})
export class FigmaCodeGeneratorComponent {
  response: any;
  aiAuery: any;
  course: any;
  code: any;
  isLoading = false;
  hideBotMessage = false;
  hummanMessage: any;
  constructor(
    private commonService : CommonServiceService, private sanitized: DomSanitizer,
    private route: Router) { }
    askAi() {
      this.response = "";
      this.code = "";
      this.hideBotMessage = true;
      this.isLoading = true;
      this.hummanMessage = this.aiAuery;
      this.commonService.askFigmaAI(this.aiAuery).subscribe(item => {
      this.response = item;
      this.isLoading = false;
    }, ((error: any) => {
      if(error.status == 200) {
        this.code = error.error.text;
        this.response = this.sanitized.bypassSecurityTrustHtml(error.error.text);
        this.isLoading = false;
      } else {
        this.isLoading = false;
        console.log("Error from flask server --->" + error);
        this.response = "Network Error occured, please try again";
      }
    }))
  }

  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
