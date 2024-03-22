import { Component } from '@angular/core';
import { CommonServiceService } from '../common-service.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-test-case-generator-ai',
  templateUrl: './test-case-generator-ai.component.html',
  styleUrl: './test-case-generator-ai.component.scss'
})
export class TestCaseGeneratorAIComponent {
  response: any;
  aiAuery: any;
  course: any;
  isLoading = false;
  hideBotMessage = false;
  errMsg: any;
  aiMessageArray: any = [];
  humanMessageArray:any = [];
  constructor(
    private commonService : CommonServiceService, private route: Router) {
     }
    askAi() {
        this.errMsg = "";
        this.hideBotMessage = true;
        this.isLoading = true;
        this.aiMessageArray.push(this.aiAuery);
        this.commonService.unitTestCase(this.aiAuery).subscribe(item => {
        this.response = item;
        this.humanMessageArray.push(item);
        this.isLoading = false;
      }, ((error: any) => {
        this.isLoading = false;
        if (error.error.text.includes("I don't know!" || "I dont Know"))
        {
          this.humanMessageArray.push("I am a unit test case generator AI, I am sorry I cant answer your question")
        } else {
          this.humanMessageArray.push(error.error.text);
          // this.response = error.error.text;
        }
        console.log("Error from flask server --->" + error);
        // this.response = "Network Error occured, please try again";
      }))
  }
  gotoHome() {
    this.route.navigateByUrl("home");
  }
}
